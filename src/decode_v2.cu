#include "cuda_runtime.h"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>

// nvcc -arch=sm_90a -std=c++17 decode_v2.cu -o test && ./test 

#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

#define BLOCK_SIZE 512
#define CLUSTER_SIZE 4

#define BATCH_SIZE 1 
#define HEAD_DIM 128    // attn head dimension
#define HEAD_NUM 32     // attn head number
#define FFN_HIDDEN 4096 // ffn hidden dimension
#define HIDDEN_DIM 4096 // token embedding dimension
#define SEQ_LEN 4096    // sequence length

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 5.0);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(1.0f);
        }   
    }   
}

__device__ half dot(
    half* A,
    half* B,
    int len 
)
{
    half res = __float2half(0.0f);
    #pragma unroll
    for (int i = 0; i < len; i++) {
        res += __hmul(A[i], B[i]);
    }   
    return res;
}

__global__ void __cluster_dims__(1, CLUSTER_SIZE, 1) decode(
    half* output, // batch * hidden_dim
    half* input,  // batch * 1 * hidden_dim
    half* w_q,    // batch * hidden_dim * head_num * head_dim
    half* w_k,    // batch * hidden_dim * head_num * head_dim
    half* w_v,    // batch * hidden_dim * head_num * head_dim
    half* w_o,    // batch * head_num * head_dim * hidden_dim
    half* k_cache,// batch * head_num * seqlen * head_dim
    half* v_cache,// batch * head_num * seqlen * head_dim
    half* ffn_gate,  // hidden_dim * ffn_hidden
    half* ffn_down,  // ffn_hidden * hidden_dim
    half* ffn_up,    // hidden_dim * ffn_hidden
    half* global,    // batch * hidden_dim  
    half* w_rms_input,// hidden_dim
    half* w_rms_attn, // hidden_dim
    float* cos,      // batch * head_dim
    float* sin      // batch * head_dim
)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t batch_id         = blockIdx.x;
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % 32; // 32 per warp
    const uint32_t warp_id = tid / 32;

    // Load input [1 x HIDDEN_DIM / CLUSTR_SIZE] to shared memory
    __shared__ __align__(16) half input_shmem[HIDDEN_DIM / CLUSTER_SIZE];
    for (int d = tid; d < HIDDEN_DIM / CLUSTER_SIZE / 2; d+=block.num_threads()) { 
        *(half2*)(&input_shmem[d * 2]) = *(half2*)(&input[batch_id * HIDDEN_DIM + d * 2]);
    }
    cluster.sync();

    // RMSNorm
    __shared__ float norm_reduction[16];
    float local_sum = 0;
    __shared__ float cluster_local_sum;
    half __align__(16) input_reg[2], weight_reg[2];
    for (int d = tid; d < HIDDEN_DIM / CLUSTER_SIZE / 2; d+=block.num_threads()) { 
        *(half2*)(&input_reg[0]) = *(half2*)(&input_shmem[batch_id * HIDDEN_DIM + d * 2]);
        *(half2*)(&weight_reg[0]) = *(half2*)(&w_rms_input[d * 2]);
        for (int di = 0; di < 2; di++)
            local_sum += __half2float(input_reg[di]) * __half2float(input_reg[di]);
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }

    if (lane_id == 0){
        norm_reduction[warp_id] = local_sum;
    }
    __syncthreads(); 
    if (tid < 16) 
        local_sum = norm_reduction[tid];
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    } 
    if (tid == 0)
        cluster_local_sum = local_sum;

    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            float* dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
            atomicAdd(dst_shmem, local_sum);
        }
    }
    cluster.sync();
    local_sum = cluster_local_sum;
    float eps = 1e-5;
    half rms_rcp = __float2half(1.f / (std::sqrt(local_sum / float(HIDDEN_DIM)) + eps));
    for (int d = tid; d < HIDDEN_DIM / CLUSTER_SIZE / 2; d+=block.num_threads()) { 
        *(half2*)(&input_reg[0]) = __hmul2(*(half2*)(&input_reg[0]), {rms_rcp, rms_rcp});
        *(half2*)(&input_shmem[d * 2]) = __hmul2(*(half2*)(&input_reg[0]), *(half2*)(&weight_reg[0]));
    }
    __syncthreads();
    // printf("%f \n", __half2float(input_shmem[0]));
    // // Compute hidden * wq
    // half __align__(16) w_qkv_reg[8];
    // half __align__(16) input_reg[8];
    // __shared__ __align__(16) half local_qkv_reduction[16 * 8];
    // __shared__ __align__(16) half local_q[HEAD_DIM / CLUSTER_SIZE];
    // __shared__ __align__(16) half local_kv[HEAD_DIM / CLUSTER_SIZE];

    // for (int d = 0; d < HEAD_DIM / CLUSTER_SIZE; d+=8) {
    //     // shared memory -> register
    //     *(uint4*)(&input_reg[0]) = *(uint4*)(&input_shmem[tid * (SEQ_LEN / block.num_threads())]);
    //     half __align__(16) local_sum[8] = {__float2half(0.0)};
    //     for (int i = 0; i < SEQ_LEN / block.num_threads(); i++) {
    //         *(uint4*)(&w_qkv_reg[0]) = *(uint4*)(&w_q[
    //                 batch_id * SEQ_LEN * HEAD_DIM * HEAD_NUM
    //                 + head_id * HEAD_DIM * SEQ_LEN
    //                 + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE)
    //                 + (tid * (SEQ_LEN / block.num_threads()) + i) * HEAD_DIM
    //                 + d
    //                 ]);
    //         // TODO: Use half2 __hmul2 but exist bug
    //         for (int di = 0; di < 8; di++) {
    //             local_sum[di] += __hmul(input_reg[i], w_qkv_reg[di]);
    //         }
    //     }
    //     // reduction in warp
    //     #pragma unroll
    //     for (int mask = 16; mask > 0; mask >>= 1) {
    //         // warp shuffle
    //         *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
    //         *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
    //         *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
    //         *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
    //     }
    //     if (lane_id == 0) {
    //         *(uint4*)(&local_qkv_reduction[warp_id * 8]) = *(uint4*)(&local_sum[0]);
    //     }
    //     __syncthreads();
    //     // reduction
    //     if (tid < 16) {
    //         *(uint4*)(&local_sum[d]) = *(uint4*)(&local_qkv_reduction[tid * 8]);
    //     }
    //     for (int mask = 8; mask > 0; mask >>= 1) {
    //         *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
    //         *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
    //         *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
    //         *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
    //     }
    //     // use the first thread to do reduction
    //     if (tid == 0)
    //         *(uint4*)(&local_q[d]) = *(uint4*)(&local_sum[0]);
    // }
    // __syncthreads();

    // // *##########################
    // // todo：RoPE

    // // *##########################

    // // Compute hidden * wk
    // for (int d = 0; d < HEAD_DIM / CLUSTER_SIZE; d+=8) {
    //     *(uint4*)(&input_reg[0]) = *(uint4*)(&input_shmem[tid * (SEQ_LEN / block.num_threads())]);
    //     half __align__(16) local_sum[8] = {__float2half(0.0)};
    //     for (int i = 0; i < SEQ_LEN / block.num_threads(); i++) {
    //         *(uint4*)(&w_qkv_reg[0]) = *(uint4*)(&w_k[batch_id * SEQ_LEN * HEAD_DIM * HEAD_NUM + head_id * HEAD_DIM * SEQ_LEN + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE) + (tid * (SEQ_LEN / block.num_threads()) + i) * HEAD_DIM + d]);
    //         // TODO: Use half2 __hmul2 but exist bug
    //         for (int di = 0; di < 8; di++) {
    //             local_sum[di] += __hmul(input_reg[i], w_qkv_reg[di]);
    //         }
    //     }
    //     #pragma unroll
    //     for (int mask = 16; mask > 0; mask >>= 1) {
    //         *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
    //         *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
    //         *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
    //         *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
    //     }
    //     if (lane_id == 0) {
    //         *(uint4*)(&local_qkv_reduction[warp_id * 8]) = *(uint4*)(&local_sum[0]);
    //     }
    //     __syncthreads();
    //     if (tid < 16) {
    //         *(uint4*)(&local_sum[d]) = *(uint4*)(&local_qkv_reduction[tid * 8]);
    //     }
    //     for (int mask = 8; mask > 0; mask >>= 1) {
    //         *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
    //         *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
    //         *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
    //         *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
    //     }
    //     if (tid == 0)
    //         *(uint4*)(&local_kv[d]) = *(uint4*)(&local_sum[0]);
    // }
    // __syncthreads();

    // // *##########################
    // // todo：RoPE

    // // *##########################

    // // Compute q * k^T
    // extern __shared__ __align__(16) uint8_t attn_weight[];
    // half *attn_weight_smem = reinterpret_cast<half*>(attn_weight);
    // half *attn_weight_smem_buffer = reinterpret_cast<half*>(attn_weight + SEQ_LEN * 2);
    // for (int d = tid; d < SEQ_LEN; d+=block.num_threads()) {
    //     attn_weight_smem[d] = __float2half(0.0f);
    //     attn_weight_smem_buffer[d] = __float2half(0.0f);
    // }
    // cluster.sync();

    // // Load K cache to register
    // half __align__(16) kv_cache_reg[8];
    // for (int d = tid; d < SEQ_LEN; d+=block.num_threads()) {
    //     for (int i = 0; i < (HEAD_DIM / CLUSTER_SIZE) / 8; i++) {
    //         *(uint4*)(&input_reg[0]) = *(uint4*)(&local_q[i * 8]);
    //         if (d != SEQ_LEN - 1) {
    //             *(uint4*)(&kv_cache_reg[0]) = *(uint4*)(&k_cache[batch_id * HEAD_DIM * HEAD_NUM * (SEQ_LEN - 1) + head_id * HEAD_DIM * (SEQ_LEN - 1) + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE) + d * HEAD_DIM + i * 8]);
    //             attn_weight_smem[d] += dot(input_reg, kv_cache_reg, 8);
    //         } else {
    //             *(uint4*)(&kv_cache_reg[0]) = *(uint4*)(&local_kv[i * 8]);
    //             attn_weight_smem[d] += dot(input_reg, kv_cache_reg, 8);
    //         }
    //     }
    // }
    // // sync everything in cluster
    // cluster.sync();

    // // Attention weight reduce through DSM (distributed shared mem)
    // for (int i = 1; i < cluster.num_blocks() - 1; i++) {
    //     __shared__ uint64_t barrier;
    //     // Load neighbor block shmem data to this block's buffer within cluster
    //     if (tid == 0) {
    //         uint32_t size = SEQ_LEN * 2;
    //         uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    //         asm volatile (
    //             "mbarrier.init.shared::cta.b64 [%0], %1;"
    //             :
    //             : "r"(bar_ptr), "r"(1)
    //         );
    //         asm volatile (
    //             "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
    //             :
    //             : "r"(bar_ptr), "r"(size)
    //         );
    //     }
    //     cluster.sync();
    //     if (tid == 0) {
    //         uint32_t size = SEQ_LEN * 2;
    //         uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    //         uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(attn_weight_smem));
    //         uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(attn_weight_smem_buffer));
    //         uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
    //         uint32_t neighbor_dst_addr;
    //         asm volatile (
    //             "mapa.shared::cluster.u32 %0, %1, %2;\n"
    //             : "=r"(neighbor_dst_addr)
    //             : "r"(dst_addr), "r"(dst_cta)
    //         );
    //         uint32_t neighbor_dst_bar;
    //         asm volatile (
    //             "mapa.shared::cluster.u32 %0, %1, %2;\n"
    //             : "=r"(neighbor_dst_bar)
    //             : "r"(bar_ptr), "r"(dst_cta)
    //         );
    //         asm volatile (
    //             "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
    //             :
    //             :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
    //             : "memory"
    //         );
    //     }
    //     uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    //     asm volatile (
    //         "{\n"
    //         ".reg .pred                P1;\n"
    //         "LAB_WAIT:\n"
    //         "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
    //         "@P1                       bra.uni DONE;\n"
    //         "bra.uni                   LAB_WAIT;\n"
    //         "DONE:\n"
    //         "}\n"
    //         :: "r"(bar_ptr),
    //         "r"(0)
    //     );

    //     // Add
    //     for (int d = tid; d < SEQ_LEN; d+=block.num_threads()) {
    //         half buffer = attn_weight_smem_buffer[d];
    //         attn_weight_smem[d] += buffer;
    //     }
    //     __syncthreads();
    // }

    // // Softmax reduce
    // float local_scale = 0.0f;
    // __shared__ float final_scale;
    // for (int i = tid; i < SEQ_LEN; i+=block.num_threads()) {
    //     half tmp = hexp(attn_weight_smem[i] / __float2half(1.0 * HEAD_DIM));
    //     attn_weight_smem[i] = tmp;
    //     local_scale += __half2float(tmp);
    // }
    // __shared__ float local_attn_weight_reduction[16];
    // #pragma unroll
    // for (int mask = 16; mask > 0; mask >>= 1) {
    //     local_scale += __shfl_down_sync(0xffffffff, local_scale, mask);
    // }
    // if (lane_id == 0)
    //     local_attn_weight_reduction[warp_id] = local_scale;
    // __syncthreads();
    // if (tid < 16)
    //     local_scale = local_attn_weight_reduction[tid];
    // #pragma unroll
    // for (int mask = 8; mask > 0; mask >>= 1) {
    //     local_scale += __shfl_down_sync(0xffffffff, local_scale, mask);
    // }
    // if(tid == 0) {
    //     final_scale = local_scale;
    // }
    // __syncthreads();
    // for (int i = tid; i < SEQ_LEN; i+=block.num_threads()) {
    //     attn_weight_smem[i] = __float2half(__half2float(attn_weight_smem[i]) / final_scale);
    // }

    // // Compute hidden * w_v
    // for (int d = 0; d < HEAD_DIM / CLUSTER_SIZE; d+=8) {
    //     *(uint4*)(&input_reg[0]) = *(uint4*)(&input_shmem[tid * (SEQ_LEN / block.num_threads())]);
    //     half __align__(16) local_sum[8] = {__float2half(0.0)};
    //     for (int i = 0; i < SEQ_LEN / block.num_threads(); i++) {
    //         *(uint4*)(&w_qkv_reg[0]) = *(uint4*)(&w_v[batch_id * SEQ_LEN * HEAD_DIM * HEAD_NUM + head_id * HEAD_DIM * SEQ_LEN + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE) + (tid * (SEQ_LEN / block.num_threads()) + i) * HEAD_DIM + d]);
    //         // TODO: Use half2 __hmul2 but exist bug
    //         for (int di = 0; di < 8; di++) {
    //             local_sum[di] += __hmul(input_reg[i], w_qkv_reg[di]);
    //         }
    //     }
    //     #pragma unroll
    //     for (int mask = 16; mask > 0; mask >>= 1) {
    //         *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
    //         *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
    //         *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
    //         *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
    //     }
    //     if (lane_id == 0) {
    //         *(uint4*)(&local_qkv_reduction[warp_id * 8]) = *(uint4*)(&local_sum[0]);
    //     }
    //     __syncthreads();
    //     if (tid < 16) {
    //         *(uint4*)(&local_sum[d]) = *(uint4*)(&local_qkv_reduction[tid * 8]);
    //     }
    //     for (int mask = 8; mask > 0; mask >>= 1) {
    //         *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
    //         *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
    //         *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
    //         *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
    //     }
    //     if (tid == 0)
    //         *(uint4*)(&local_kv[d]) = *(uint4*)(&local_sum[0]);
    // }
    // __syncthreads();

    // // Compute attn_weight * v
    // half __align__(16) local_v_reg[8];
    // __shared__ __align__(16) half local_output[HEAD_DIM / CLUSTER_SIZE];
    // __shared__ __align__(16) half output_reduction[16 * 8];
    // for (int d = 0; d < HEAD_DIM / CLUSTER_SIZE; d+=8) {
    //     *(uint4*)(&input_reg[0]) = *(uint4*)(&attn_weight_smem[tid * (SEQ_LEN / block.num_threads())]);
    //     half __align__(16) local_sum[8] = {__float2half(0.0)};
    //     for (int i = 0; i < SEQ_LEN / block.num_threads(); i++) {
    //         if (tid * (SEQ_LEN / block.num_threads()) + i == SEQ_LEN - 1)
    //             *(uint4*)(&local_v_reg[0]) = *(uint4*)(&local_kv[d]);
    //         else
    //             *(uint4*)(&local_v_reg[0]) = *(uint4*)(&v_cache[batch_id * HEAD_DIM * HEAD_NUM * (SEQ_LEN - 1) + head_id * HEAD_DIM * (SEQ_LEN - 1) + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE) + (tid * (SEQ_LEN / block.num_threads()) + i) * HEAD_DIM + d]);
    //         for (int di = 0; di < 8; di++) {
    //             local_sum[di] += __hmul(input_reg[i], local_v_reg[di]);
    //         }
    //     }
    //     #pragma unroll
    //     for (int mask = 16; mask > 0; mask >>= 1) {
    //         *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
    //         *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
    //         *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
    //         *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
    //     }
    //     if (lane_id == 0) {
    //         *(uint4*)(&output_reduction[warp_id * 8]) = *(uint4*)(&local_sum[0]);
    //     }
    //     __syncthreads();
    //     if (tid < 16) {
    //         *(uint4*)(&local_sum[d]) = *(uint4*)(&output_reduction[tid * 8]);
    //     }
    //     for (int mask = 8; mask > 0; mask >>= 1) {
    //         *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
    //         *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
    //         *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
    //         *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
    //     }
    //     if (tid == 0)
    //         *(uint4*)(&local_output[d]) = *(uint4*)(&local_sum[0]);
    // }
    // __syncthreads();

    // // Compute output * w_o
    // half w_o_reg;
    // half local_output_reg;
    // // printf("%f \n", __half2float(global[0]));
    // for (int i = tid; i < SEQ_LEN; i+=block.num_threads()) {
    //     half local_sum = __float2half(0.0);
    //     for (int j = 0; j < HEAD_DIM / CLUSTER_SIZE; j++) {
    //         local_output_reg = local_output[j];
    //         w_o_reg = w_o[head_id * HEAD_DIM * SEQ_LEN + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE) * SEQ_LEN + i + j * SEQ_LEN];
    //         local_sum += __hmul(local_output_reg, w_o_reg);
    //     }
    //     // Exists bug here
    //     atomicAdd(&global[i], local_sum);
    // }

    // // Load input to shared
    // for (int d = tid; d < SEQ_LEN / 8; d+=block.num_threads()) {
    //     *(uint4*)(&input_shmem[d * 8]) = *(uint4*)(&global[batch_id * SEQ_LEN + d * 8]);
    // }
    // __syncthreads();

    // // *##########################
    // // todo：RMSNorm

    // // *##########################

    // // Compute FFN1
    // half __align__(16) w_ffn1_reg[8];
    // __shared__ __align__(16) half local_output_reduction[16 * 8];
    // for (int d = 0; d < HEAD_DIM / CLUSTER_SIZE; d+=8) {
    //     *(uint4*)(&input_reg[0]) = *(uint4*)(&input_shmem[tid * (SEQ_LEN / block.num_threads())]);
    //     half __align__(16) local_sum[8] = {__float2half(0.0)};
    //     for (int i = 0; i < SEQ_LEN / block.num_threads(); i++) {
    //         *(uint4*)(&w_ffn1_reg[0]) = *(uint4*)(&ffn_1[batch_id * SEQ_LEN * SEQ_LEN + head_id * HEAD_DIM * SEQ_LEN + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE) + (tid * (SEQ_LEN / block.num_threads()) + i) * HEAD_DIM + d]);
    //         for (int di = 0; di < 8; di++) {
    //             local_sum[di] += __hmul(input_reg[i], w_ffn1_reg[di]);
    //         }
    //     }
    //     #pragma unroll
    //     for (int mask = 16; mask > 0; mask >>= 1) {
    //         *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
    //         *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
    //         *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
    //         *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
    //     }
    //     if (lane_id == 0) {
    //         *(uint4*)(&local_output_reduction[warp_id * 8]) = *(uint4*)(&local_sum[0]);
    //     }
    //     __syncthreads();

    //     if (tid < 16) {
    //         *(uint4*)(&local_sum[d]) = *(uint4*)(&local_output_reduction[tid * 8]);
    //     }
    //     for (int mask = 8; mask > 0; mask >>= 1) {
    //         *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
    //         *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
    //         *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
    //         *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
    //     }

    //     // *##########################
    //     // todo: use SiLU 

    //     // *##########################

    //     if (tid == 0) {
    //         *(uint4*)(&local_output[d]) = *(uint4*)(&local_sum[0]);
    //         for (int di = 0; di < 8; di++) {
    //             local_output[di] = __hmax(local_output[di], __float2half(0.0));
    //         }
    //     }
    // }
    // __syncthreads();

    // // Compute FFN2
    // half w_ffn2_reg;
    // for (int i = tid; i < SEQ_LEN; i+=block.num_threads()) {
    //     half local_sum = __float2half(0.0);
    //     for (int j = 0; j < HEAD_DIM / CLUSTER_SIZE; j++) {
    //         local_output_reg = local_output[j];
    //         w_ffn2_reg = ffn_2[batch_id * SEQ_LEN * SEQ_LEN + head_id * HEAD_DIM * SEQ_LEN + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE) * SEQ_LEN + i + j * SEQ_LEN];
    //         local_sum += __hmul(local_output_reg, w_ffn2_reg);
    //     }
    //     // Exists bug here
    //     atomicAdd(&output[i], local_sum);
    // }
}

int main(int argc, char** argv) {
    // shared memory size per threadBlock
    cudaFuncSetAttribute(decode, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(float) * BATCH_SIZE * HIDDEN_DIM * 5);
    // at most 16 blocks per cluster
    cudaFuncSetAttribute(decode, cudaFuncAttributeNonPortableClusterSizeAllowed, 16);

    half *h_input, *d_input;
    half *h_k_cache, *d_k_cache;
    half *h_v_cache, *d_v_cache;
    half *h_w_q, *d_w_q;
    half *h_w_k, *d_w_k;
    half *h_w_v, *d_w_v;
    half *h_w_o, *d_w_o;
    half *h_ffn_gate, *d_ffn_gate;
    half *h_ffn_down, *d_ffn_down;
    half *h_ffn_up, *d_ffn_up;
    half *h_rms_input, *d_rms_input;
    half *h_rms_attn, *d_rms_attn;
    float *h_cos, *d_cos;
    float *h_sin, *d_sin;
    h_input = new half[BATCH_SIZE * 1 * HIDDEN_DIM];
    h_w_q = new half[BATCH_SIZE * HEAD_NUM * HIDDEN_DIM * HEAD_DIM];
    h_w_k = new half[BATCH_SIZE * HEAD_NUM * HIDDEN_DIM * HEAD_DIM];
    h_w_v = new half[BATCH_SIZE * HEAD_NUM * HIDDEN_DIM * HEAD_DIM];
    h_w_o = new half[BATCH_SIZE * HEAD_NUM * HEAD_DIM * HIDDEN_DIM];
    h_k_cache = new half[BATCH_SIZE * HEAD_NUM * (SEQ_LEN - 1) * HEAD_DIM];
    h_v_cache = new half[BATCH_SIZE * HEAD_NUM * (SEQ_LEN - 1) * HEAD_DIM];
    h_ffn_gate = new half[HIDDEN_DIM * FFN_HIDDEN];
    h_ffn_up = new half[HIDDEN_DIM * FFN_HIDDEN];
    h_ffn_down = new half[FFN_HIDDEN * HIDDEN_DIM];
    h_rms_input = new half[HIDDEN_DIM];
    h_rms_attn = new half[HIDDEN_DIM];
    h_cos = new float[BATCH_SIZE * HEAD_DIM];
    h_sin = new float[BATCH_SIZE * HEAD_DIM];

    fill_matrix(h_input, BATCH_SIZE * 1 * HIDDEN_DIM);
    fill_matrix(h_w_q, HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    fill_matrix(h_w_k, HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    fill_matrix(h_w_v, HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    fill_matrix(h_w_o, HEAD_NUM * HEAD_DIM * HIDDEN_DIM);
    fill_matrix(h_k_cache, BATCH_SIZE * HEAD_NUM * (SEQ_LEN - 1) * HEAD_DIM);
    fill_matrix(h_v_cache, BATCH_SIZE * HEAD_NUM * (SEQ_LEN - 1) * HEAD_DIM);
    fill_matrix(h_ffn_gate, HIDDEN_DIM * FFN_HIDDEN);
    fill_matrix(h_ffn_down, FFN_HIDDEN * HIDDEN_DIM);
    fill_matrix(h_ffn_up, HIDDEN_DIM * FFN_HIDDEN);
    fill_matrix(h_rms_input, HIDDEN_DIM);
    fill_matrix(h_rms_attn, HIDDEN_DIM);

    // Init cos, sin used in RoPE
    int encode_point_offset = 0;
    float rope_scale = 1;
    float rope_theta = 500000;
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            float inv_freq =(encode_point_offset / rope_scale) / (std::pow(rope_theta, float(2 * (j % (HEAD_DIM / 2))) / float(HEAD_DIM)));
            h_cos[i * HEAD_DIM + j] = std::cos(inv_freq);
            h_sin[i * HEAD_DIM + j] = std::sin(inv_freq);
        }
    }

    cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * BATCH_SIZE * 1 * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_q), sizeof(half) * BATCH_SIZE * HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_k), sizeof(half) * BATCH_SIZE * HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_v), sizeof(half) * BATCH_SIZE * HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_o), sizeof(half) * BATCH_SIZE * HEAD_NUM * HEAD_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_k_cache), sizeof(half) * BATCH_SIZE * HEAD_NUM * (SEQ_LEN - 1) * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_v_cache), sizeof(half) * BATCH_SIZE * HEAD_NUM * (SEQ_LEN - 1) * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_gate), sizeof(half) * HIDDEN_DIM * FFN_HIDDEN);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_down), sizeof(half) * FFN_HIDDEN * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_up), sizeof(half) * HIDDEN_DIM * FFN_HIDDEN);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_input), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_attn), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_cos), sizeof(float) * BATCH_SIZE * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_sin), sizeof(float) * BATCH_SIZE * HEAD_DIM);

    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * BATCH_SIZE * 1 * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_q), h_w_q, sizeof(half) * HEAD_NUM * HIDDEN_DIM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_k), h_w_k, sizeof(half) * HEAD_NUM * HIDDEN_DIM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_v), h_w_v, sizeof(half) * HEAD_NUM * HIDDEN_DIM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_o), h_w_o, sizeof(half) * HEAD_NUM * HEAD_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_k_cache), h_k_cache, sizeof(half) * BATCH_SIZE * HEAD_NUM * (SEQ_LEN - 1) * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_v_cache), h_v_cache, sizeof(half) * BATCH_SIZE * HEAD_NUM * (SEQ_LEN - 1) * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_gate), h_ffn_gate, sizeof(half) * HIDDEN_DIM * FFN_HIDDEN, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_down), h_ffn_down, sizeof(half) * FFN_HIDDEN * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_up), h_ffn_up, sizeof(half) * HIDDEN_DIM * FFN_HIDDEN, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_input), h_rms_input, sizeof(half) * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_attn), h_rms_attn, sizeof(half) * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_cos), h_cos, sizeof(float) * BATCH_SIZE * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_sin), h_sin, sizeof(float) * BATCH_SIZE * HEAD_DIM, cudaMemcpyHostToDevice);

    half* h_output, *d_output;
    h_output = new half[BATCH_SIZE * HEAD_NUM * HIDDEN_DIM];
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * BATCH_SIZE * HEAD_NUM * HIDDEN_DIM);

    half h_global[BATCH_SIZE * HIDDEN_DIM] = {__float2half(0.0)};
    half *global_reduce;
    cudaMalloc(reinterpret_cast<void**>(&global_reduce), sizeof(half) * BATCH_SIZE * HIDDEN_DIM);
    cudaMemcpy(reinterpret_cast<void*>(global_reduce), h_global, sizeof(half) * BATCH_SIZE * HIDDEN_DIM, cudaMemcpyHostToDevice);

    dim3 grid(BATCH_SIZE, HEAD_NUM * CLUSTER_SIZE); // 32 * 4
    dim3 block(BLOCK_SIZE); // 512

    int wmup = 50;
    int test = 10;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    for (int i = 0; i < wmup; i++) {
        decode<<<grid, block, sizeof(float) * BATCH_SIZE * HIDDEN_DIM * 5>>>(
            d_output,
            d_input,
            d_w_q,
            d_w_k,
            d_w_v,
            d_w_o,
            d_k_cache,
            d_v_cache,
            d_ffn_gate,
            d_ffn_down,
            d_ffn_up,
            global_reduce,
            d_rms_input,
            d_rms_attn,
            d_cos,
            d_sin
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < test; i++) {
        decode<<<grid, block, sizeof(float) * BATCH_SIZE * HIDDEN_DIM * 5>>>(
            d_output,
            d_input,
            d_w_q,
            d_w_k,
            d_w_v,
            d_w_o,
            d_k_cache,
            d_v_cache,
            d_ffn_gate,
            d_ffn_down,
            d_ffn_up,
            global_reduce,
            d_rms_input,
            d_rms_attn,
            d_cos,
            d_sin
        );
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << (ms / (1.0 * test)) * 1e3 << " us" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * BATCH_SIZE * HEAD_NUM * HIDDEN_DIM, cudaMemcpyDeviceToHost);
    return 0;
}
