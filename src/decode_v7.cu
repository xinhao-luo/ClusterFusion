#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <cuda.h>    
#include <cuda/barrier>
#include <cudaTypedefs.h>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
#include <iostream>
#include <random>
#include <stdio.h>

// nvcc --generate-code=arch=compute_90a,code=sm_90a -O3 -std=c++17 -lcuda decode_v7.cu -o test && ./test

#define HEAD_DIM 128    // attn head dimension
#define HEAD_NUM 32     // attn head number
#define FFN_DIM 4096 // ffn hidden dimension
#define HIDDEN_DIM 4096 // token embedding dimension
#define SEQ_LEN 4096    // sequence length

#define NUM_WARPS 4 // 4 8 16 32
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE) // 512
#define CLUSTER_SIZE 4 // 2 4 8 16
#define NUM_PER_THREAD 8
#define NUM_ROW_PER_WARP (HEAD_DIM / NUM_WARPS) // 32
#define NUM_THREAD_PER_ROW (WARP_SIZE / NUM_ROW_PER_WARP) // 1
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW) // 8
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE) // 1024
#define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) // 1024
#define FFN_DIM_PER_BLOCK (FFN_DIM / CLUSTER_SIZE) // 1024

#define NUM_ROW_PER_WARP_3 (DIM_PER_BLOCK / NUM_WARPS) // 256
#define NUM_ROW_PER_WARP_4 (FFN_DIM_PER_BLOCK / NUM_WARPS) // 256

#define TMA_LOAD_ONCE 64 // 8 16 32 64 128 256

#define NUM_ROW_PER_WARP_2 (TMA_LOAD_ONCE / NUM_WARPS) // 16
#define NUM_THREAD_PER_ROW_2 (WARP_SIZE / NUM_ROW_PER_WARP_2) // 2
#define NUM_PER_ROW_2 (NUM_PER_THREAD * NUM_THREAD_PER_ROW_2) // 16

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 5.0);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(0.01f);
        }   
    }   
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) single_decode(
    half* output, // 1 * hidden_dim
    half* input,  // 1 * hidden_dim
    half* w_qkv,    // 3 * hidden_dim * hidden_dim
    half* kv_cache,// 2 * seqlen * head_num * head_dim
    half* ffn_gate,  // hidden_dim * ffn_dim
    half* ffn_down,  // ffn_dim * hidden_dim
    half* ffn_up,    // hidden_dim * ffn_dim
    half* global_reduce,    // hidden_dim  
    half* w_rms_input,// hidden_dim
    half* w_rms_attn, // hidden_dim
    float* cos,       // head_dim
    float* sin,       // head_dim
    const __grid_constant__ CUtensorMap tensor_map,
    const __grid_constant__ CUtensorMap tensor_map_kv_cache,
    const __grid_constant__ CUtensorMap tensor_map_weight_o
)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank() % HEAD_NUM;
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; // 32 per warp
    const uint32_t warp_id = tid / WARP_SIZE;

    // Init shared memory
    __shared__ __align__(16) half input_shmem[DIM_PER_BLOCK];
    __shared__ float reduction[NUM_WARPS];
    __shared__ float cluster_local_sum;
    __shared__ __align__(16) half local_q[HEAD_DIM];
    __shared__ __align__(16) half local_kv[HEAD_DIM];
    __shared__ alignas(128) half weight[TMA_LOAD_ONCE * HEAD_DIM];
    __shared__ alignas(128) half weight_buffer[TMA_LOAD_ONCE * HEAD_DIM];
    __shared__ __align__(16) half local_buffer[HEAD_DIM];
    __shared__ __align__(16) half attn_weight[KV_DIM_PER_BLOCK];
    // Initialize shared memory barrier with the number of threads participating in the barrier.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    __shared__ barrier bar_buffer;
    __shared__ uint64_t barrier;

    // Load input [1 x HIDDEN_DIM / CLUSTR_SIZE] to shared memory
    #pragma unroll
    for (int i = tid * 8; i < DIM_PER_BLOCK; i+=BLOCK_SIZE * 8) {
        *(uint4*)(&input_shmem[i]) = *(uint4*)(&input[cluster_block_id * DIM_PER_BLOCK + i]);
    }
    block.sync();

    // RMSNorm
    float local_sum = 0;
    half __align__(16) reg_input_norm[2], reg_weight_norm[2];
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
        for (int di = 0; di < 2; di++)
            local_sum += __half2float(reg_input_norm[di] * reg_input_norm[di]);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }
    if (lane_id == 0){
        reduction[warp_id] = local_sum;
    }
    block.sync(); 
    if (tid < NUM_WARPS) 
        local_sum = reduction[tid];
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    } 
    if (tid == 0)
        cluster_local_sum = local_sum;
    cluster.sync();
    // Reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            float* dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
            atomicAdd(dst_shmem, local_sum);
        }
        cluster.sync();
    }
    float eps = 1e-5;
    half rms_rcp = __float2half(1.f / (std::sqrt(cluster_local_sum / float(HIDDEN_DIM)) + eps));
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
        *(half2*)(&reg_input_norm[0]) = __hmul2(*(half2*)(&reg_input_norm[0]), {rms_rcp, rms_rcp});
        *(half2*)(&reg_weight_norm[0]) = *(half2*)(&w_rms_input[d]);
        *(half2*)(&input_shmem[d]) = __hmul2(*(half2*)(&reg_input_norm[0]), *(half2*)(&reg_weight_norm[0]));
    }
    block.sync();

    // Compute hidden @ wq
    float tmp = 0.0;
    half __align__(16) reg_input[NUM_PER_THREAD];
    half __align__(16) reg_weight[NUM_PER_THREAD];
    if (tid == 0) {
        // Initialize barrier. All `blockDim.x` threads in block participate.
        init(&bar, blockDim.x);
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
        init(&bar_buffer, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    // Syncthreads so initialized barrier is visible to all threads.
    block.sync();

    // Preload weight_q
    barrier::arrival_token token[2];
    if (tid == 0) {
        // Initiate bulk tensor copy.
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK, bar);
        // Arrive on the barrier and tell how many bytes are expected to come in.
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        // Other threads just arrive.
        token[0] = bar.arrive();
    }

    // Compute input @ w_q
    uint input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d] * weight[(input_idx + i + d) * HEAD_DIM + weight_idx]);
                }
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d] * weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]);
                }
            }
        }
    }
    bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_q[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // Preload weight_k
    tmp = 0.0;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    // Compute input @ w_k
    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d] * weight[(input_idx + i + d) * HEAD_DIM + weight_idx]);
                }
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d] * weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]);
                }
            }
        }
    }
    bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_kv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();
    // if(tid == 0)
    //     printf("%f, %f \n", __half2float(local_q[0]), __half2float(local_kv[127]));

    // Compute partial RoPE
    half2 q_rope, q_rope_1;
    half2 k_rope, k_rope_1;
    float2 cos_reg, sin_reg;
    if (tid < HEAD_DIM / 2) {
        q_rope = *(half2*)(&local_q[tid * 2]);
        k_rope = *(half2*)(&local_kv[tid * 2]);
        if (tid * 2 < HEAD_DIM / 2) {
            q_rope_1 = *(half2*)(&local_q[HEAD_DIM / 2 + tid * 2]);
            k_rope_1 = *(half2*)(&local_kv[HEAD_DIM / 2 + tid * 2]);
            cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
            sin_reg = {-sin[HEAD_DIM / 2 + tid * 2], -sin[HEAD_DIM / 2 + tid * 2 + 1]};
        } else {
            q_rope_1 = *(half2*)(&local_q[tid * 2 - HEAD_DIM / 2]);
            k_rope_1 = *(half2*)(&local_kv[tid * 2 - HEAD_DIM / 2]);
            cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
            sin_reg = {sin[tid * 2 - HEAD_DIM / 2], sin[tid * 2 + 1 - HEAD_DIM / 2]};
        }
        *(half2*)(&local_q[tid * 2]) = __hadd2(__hmul2(q_rope, __float22half2_rn(cos_reg)), __hmul2(q_rope_1, __float22half2_rn(sin_reg)));
        *(half2*)(&local_kv[tid * 2]) = __hadd2(__hmul2(k_rope, __float22half2_rn(cos_reg)), __hmul2(k_rope_1, __float22half2_rn(sin_reg)));
    }

    uint32_t size = HEAD_DIM * sizeof(half);
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));

    // Q reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        // Load neighbor block shmem data to this block's buffer within cluster
        if (tid == 0) {
            asm volatile (
                "mbarrier.init.shared::cta.b64 [%0], %1;"
                :
                : "r"(bar_ptr), "r"(1)
            );
            asm volatile (
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(bar_ptr), "r"(size)
            );
        }
        cluster.sync();
        if (tid == 0) {
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_q));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );

        // Add
        if (tid < HEAD_DIM / 2) {
            half2 buffer = *(half2*)(&local_buffer[tid * 2]);
            *(half2*)(&local_q[tid * 2]) = __hadd2(*(half2*)(&local_q[tid * 2]), buffer);
        }
        cluster.sync();
    }
    
    // K reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        // Load neighbor block shmem data to this block's buffer within cluster
        if (tid == 0) {
            asm volatile (
                "mbarrier.init.shared::cta.b64 [%0], %1;"
                :
                : "r"(bar_ptr), "r"(1)
            );
            asm volatile (
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(bar_ptr), "r"(size)
            );
        }
        cluster.sync();
        if (tid == 0) {
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_kv));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );

        // Add
        if (tid < HEAD_DIM / 2) {
            half2 buffer = *(half2*)(&local_buffer[tid * 2]);
            *(half2*)(&local_kv[tid * 2]) = __hadd2(*(half2*)(&local_kv[tid * 2]), buffer);
        }
        cluster.sync();
    }
    // if(head_id == 0 && cluster_block_id == CLUSTER_SIZE - 1 && tid == 0)
    //     printf("%f, %f, %f, %f \n", __half2float(local_q[0]), __half2float(local_kv[127]), __half2float(local_q[127]), __half2float(local_kv[0]));
    // Compute Q @ K^T
    input_idx = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    weight_idx = warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2;
    tmp = 0.0;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_kv_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    // Compute Q @ K^T
    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map_kv_cache, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            tmp = 0.0;
            for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
                *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[weight_idx * HEAD_DIM + input_idx + j]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d] * reg_weight[d]);
                }
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                tmp += __shfl_down_sync(0xffffffff, tmp, mask);
            }
            if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
                attn_weight[warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2 + (id - 1) * TMA_LOAD_ONCE] = __float2half(tmp);
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_kv_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            tmp = 0.0;
            for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
                *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight_buffer[weight_idx * HEAD_DIM + input_idx + j]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d] * reg_weight[d]);
                }
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                tmp += __shfl_down_sync(0xffffffff, tmp, mask);
            }
            if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
                attn_weight[warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2 + (id - 1) * TMA_LOAD_ONCE] = __float2half(tmp);
            }
        }
    }
    bar_buffer.wait(std::move(token[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    tmp = 0.0;
    for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
        if (cluster_block_id == CLUSTER_SIZE - 1 && warp_id == NUM_WARPS - 1 && lane_id / NUM_THREAD_PER_ROW_2 == NUM_ROW_PER_WARP_2 - 1) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_kv[input_idx + j]);
        } else {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight_buffer[weight_idx * HEAD_DIM + input_idx + j]);
        }
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * reg_weight[d]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        attn_weight[warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2 + (KV_DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) * TMA_LOAD_ONCE] = __float2half(tmp);
    }
    block.sync();
    // // if(head_id == 0 && cluster_block_id == 0)
    // //     printf("%f, %f \n", __half2float(attn_weight[0]), __half2float(attn_weight[KV_DIM_PER_BLOCK - 1]));
    // Softmax
    float local_scale = 0.0f;
    half tmp_;
    for (int i = 0; i < KV_DIM_PER_BLOCK / BLOCK_SIZE; i++) {
        tmp_ = hexp(attn_weight[tid * (KV_DIM_PER_BLOCK / BLOCK_SIZE) + i] / __float2half(sqrt(1.0 * HEAD_DIM)));
        attn_weight[tid * (KV_DIM_PER_BLOCK / BLOCK_SIZE) + i] = tmp_;
        local_scale += __half2float(tmp_);
    }
    // if(head_id == 0 && cluster_block_id == 0)
    //     printf("%f, %f \n", __half2float(attn_weight[0]), __half2float(attn_weight[KV_DIM_PER_BLOCK - 1]));
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_scale += __shfl_down_sync(0xffffffff, local_scale, mask);
    }
    if (lane_id == 0) {
        reduction[warp_id] = local_scale;
    }
    block.sync();
    if (tid < NUM_WARPS) {
        local_scale = reduction[tid];
    }
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
        local_scale += __shfl_down_sync(0xffffffff, local_scale, mask);
    }
    if(tid == 0) {
        cluster_local_sum = local_scale;
    }
    cluster.sync();
    // Reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_scale = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            float* dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
            atomicAdd(dst_shmem, local_scale);
        }
        cluster.sync();
    }
    // if(head_id == 0 && cluster_block_id == 0)
    //     printf("%f, %f \n", __half2float(attn_weight[0]), __half2float(attn_weight[KV_DIM_PER_BLOCK - 1]));
    for (int i = tid; i < KV_DIM_PER_BLOCK; i+=BLOCK_SIZE) {
        attn_weight[i] = __float2half(__half2float(attn_weight[i]) / cluster_local_sum);
    }
    block.sync();
    // if(head_id == 0 && cluster_block_id == CLUSTER_SIZE - 1)
    //     printf("%f, %f \n", __half2float(attn_weight[0]), __half2float(attn_weight[KV_DIM_PER_BLOCK - 1]));
    
    input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    // Preload weight_v
    tmp = 0.0;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM * 2 + cluster_block_id * DIM_PER_BLOCK, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    // Compute input @ w_v
    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM * 2 + cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d] * weight[(input_idx + i + d) * HEAD_DIM + weight_idx]);
                }
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM * 2 + cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d] * weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]);
                }
            }
        }
    }
    bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_kv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }

    // V reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            asm volatile (
                "mbarrier.init.shared::cta.b64 [%0], %1;"
                :
                : "r"(bar_ptr), "r"(1)
            );
            asm volatile (
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(bar_ptr), "r"(size)
            );
        }
        cluster.sync();
        if (tid == 0) {
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_kv));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );

        // Add
        if (tid < HEAD_DIM / 2) {
            half2 buffer = *(half2*)(&local_buffer[tid * 2]);
            *(half2*)(&local_kv[tid * 2]) = __hadd2(*(half2*)(&local_kv[tid * 2]), buffer);
        }
        cluster.sync();
    }
    // if(cluster_block_id == 0 && head_id == 0)
    //     printf("%f, %f \n", __half2float(attn_weight[0]), __half2float(attn_weight[KV_DIM_PER_BLOCK - 1]));
    
    // __shared__ __align__(16) half local_q[HEAD_DIM];
    
    // Preload V
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_kv_cache, head_id * HEAD_DIM, SEQ_LEN + cluster_block_id * KV_DIM_PER_BLOCK, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    tmp = 0.0;
    // Compute attn_weight @ V
    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map_kv_cache, head_id * HEAD_DIM, SEQ_LEN + cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&attn_weight[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d] * weight[(input_idx + i + d) * HEAD_DIM + weight_idx]);
                }
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_kv_cache, head_id * HEAD_DIM, SEQ_LEN + cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&attn_weight[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d] * weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]);
                }
            }
        }
    }
    bar_buffer.wait(std::move(token[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&attn_weight[input_idx + ((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        if (cluster_block_id == CLUSTER_SIZE - 1 && i == TMA_LOAD_ONCE - NUM_PER_ROW && (lane_id % NUM_THREAD_PER_ROW) == NUM_THREAD_PER_ROW - 1) {
            weight_buffer[(input_idx + i + NUM_PER_THREAD - 1) * HEAD_DIM + weight_idx] = local_kv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW];
        }
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_q[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();
    // if(head_id == 0 && cluster_block_id == 0)
    //     printf("%f, %f \n", __half2float(local_q[0]), __half2float(local_q[127]));
    // output reduce throught DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            asm volatile (
                "mbarrier.init.shared::cta.b64 [%0], %1;"
                :
                : "r"(bar_ptr), "r"(1)
            );
            asm volatile (
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(bar_ptr), "r"(size)
            );
        }
        cluster.sync();
        if (tid == 0) {
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_q));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );

        // Add
        if (tid < HEAD_DIM / 2) {
            half2 buffer = *(half2*)(&local_buffer[tid * 2]);
            *(half2*)(&local_q[tid * 2]) = __hadd2(*(half2*)(&local_q[tid * 2]), buffer);
        }
        cluster.sync();
    }
    // if(head_id == 0 && cluster_block_id == 0)
    //     printf("%f, %f \n", __half2float(local_q[0]), __half2float(local_q[127]));
    // __shared__ alignas(128) half weight_o[HEAD_DIM][TMA_LOAD_ONCE];
    // __shared__ alignas(128) half weight_o_buffer[HEAD_DIM][TMA_LOAD_ONCE];
    // Compute output @ w_o
    // weight_idx = head_id * HEAD_DIM + cluster_block_id * DIM_PER_BLOCK * HIDDEN_DIM + warp_id * NUM_ROW_PER_WARP_3 * HIDDEN_DIM + (lane_id / NUM_THREAD_PER_ROW) * HIDDEN_DIM + (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    // for (int i = 0; i < NUM_ROW_PER_WARP_3; i+=NUM_ROW_PER_WARP) {
    //     tmp = 0.0;
    //     for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW) {
    //         *(uint4*)(&reg_input[0][0]) = *(uint4*)(&local_q[input_idx + j]);
    //         *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&w_o[weight_idx + i * HIDDEN_DIM + j]);
    //         #pragma unroll
    //         for (int d = 0; d < NUM_PER_THREAD; d++) {
    //             tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
    //         }
    //     }
    //     #pragma unroll
    //     for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
    //         tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    //     }
    //     if (lane_id % NUM_THREAD_PER_ROW == 0) {
    //         atomicAdd(&global_reduce[cluster_block_id * DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP_3 + i + lane_id / NUM_THREAD_PER_ROW], __float2half(tmp));
    //     }
    // }
    // cluster.sync();
    // printf("%lu \n", sizeof(weight_o));
    // if (tid == 0) {
    //     cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_weight_o, cluster_block_id * DIM_PER_BLOCK, head_id * HEAD_DIM, bar);
    //     token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight_o));
    // } else {
    //     token[0] = bar.arrive();
    // }

    // // Compute output @ w_o
    // for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
    //     if (id % 2) {
    //         if (tid == 0) {
    //             cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_o_buffer, &tensor_map_weight_o, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, head_id * HEAD_DIM, bar_buffer);
    //             token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_o_buffer));
    //         } else {
    //             token[id % 2] = bar_buffer.arrive();
    //         }
    //         bar.wait(std::move(token[(id - 1) % 2]));
    //         tmp = 0.0;
    //         for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
    //             *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
    //             *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight_o[input_idx + j][weight_idx]);
    //             #pragma unroll
    //             for (int d = 0; d < NUM_PER_THREAD; d++) {
    //                 tmp += __half2float(reg_input[d] * reg_weight[d]);
    //             }
    //         }
    //         #pragma unroll
    //         for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
    //             tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    //         }
    //         if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
    //             // atomicAdd(&global_reduce[cluster_block_id * DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    //         }
    //     } else {
    //         if (tid == 0) {
    //             cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_o, &tensor_map_weight_o, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, head_id * HEAD_DIM, bar);
    //             token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight_o));
    //         } else {
    //             token[id % 2] = bar.arrive();
    //         }
    //         bar_buffer.wait(std::move(token[(id - 1) % 2]));
    //         tmp = 0.0;
    //         for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
    //             *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
    //             *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight_o_buffer[input_idx + j][weight_idx]);
    //             #pragma unroll
    //             for (int d = 0; d < NUM_PER_THREAD; d++) {
    //                 tmp += __half2float(reg_input[d] * reg_weight[d]);
    //             }
    //         }
    //         #pragma unroll
    //         for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
    //             tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    //         }
    //         if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
    //             // atomicAdd(&global_reduce[cluster_block_id * DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    //         }
    //     }
    // }
    // bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    // tmp = 0.0;
    // for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
    //     *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
    //     *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight_o[input_idx + j][weight_idx]);
    //     #pragma unroll
    //     for (int d = 0; d < NUM_PER_THREAD; d++) {
    //         tmp += __half2float(reg_input[d] * reg_weight[d]);
    //     }
    // }
    // #pragma unroll
    // for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
    //     tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    // }
    // if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
    //     // atomicAdd(&global_reduce[cluster_block_id * DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2 + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    // }
    cluster.sync();

    // // Fused residual and RMSNorm
    // local_sum = 0.0;
    // cluster_local_sum = 0.0;
    // for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
    //     *(half2*)(&reg_input_norm[0]) = __hadd2(*(half2*)(&input_shmem[d]), *(half2*)(&global_reduce[cluster_block_id * DIM_PER_BLOCK + d]));
    //     *(half2*)(&input_shmem[d]) = *(half2*)(&reg_input_norm[0]);
    //     for (int di = 0; di < 2; di++)
    //         local_sum += __half2float(reg_input_norm[di] * reg_input_norm[di]);
    // }
    // #pragma unroll
    // for (int mask = 16; mask > 0; mask >>= 1) {
    //     local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    // }
    // if (lane_id == 0){
    //     reduction[warp_id] = local_sum;
    // }
    // block.sync(); 
    // if (tid < NUM_WARPS){
    //     local_sum = reduction[tid];
    // }
    // #pragma unroll
    // for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
    //     local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    // } 
    // if (tid == 0)
    //     cluster_local_sum = local_sum;
    // cluster.sync();
    // // Reduce through DSM
    // for (int i = 1; i < cluster.num_blocks() - 1; i++) {
    //     if (tid == 0) {
    //         local_sum = cluster_local_sum;
    //         int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
    //         float* dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
    //         atomicAdd(dst_shmem, local_sum);
    //     }
    //     cluster.sync();
    // }
    // rms_rcp = __float2half(1.f / (std::sqrt(cluster_local_sum / float(HIDDEN_DIM)) + eps));
    // for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
    //     *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
    //     *(half2*)(&reg_input_norm[0]) = __hmul2(*(half2*)(&reg_input_norm[0]), {rms_rcp, rms_rcp});
    //     *(half2*)(&reg_weight_norm[0]) = *(half2*)(&w_rms_attn[d]);
    //     *(half2*)(&input_shmem[d]) = __hmul2(*(half2*)(&reg_input_norm[0]), *(half2*)(&reg_weight_norm[0]));
    // }
    // block.sync();

    // // Compute gate proj
    // weight_idx = head_id * HEAD_DIM * FFN_DIM + cluster_block_id * FFN_DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP * FFN_DIM + (lane_id / NUM_THREAD_PER_ROW) * FFN_DIM + (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    // tmp = 0.0;
    // for (int i = 0; i < FFN_DIM_PER_BLOCK; i+=NUM_PER_ROW) { // 16
    //     *(uint4*)(&reg_input[0][0]) = *(uint4*)(&input_shmem[input_idx + i]);
    //     *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&ffn_gate[weight_idx + i]);
    //     #pragma unroll
    //     for (int d = 0; d < NUM_PER_THREAD; d++) {
    //         tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
    //     }
    // }
    // #pragma unroll
    // for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
    //     tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    // }
    // if (lane_id % NUM_THREAD_PER_ROW == 0) {
    //     local_q[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    // }
    // block.sync();

    // // gate proj reduce through DSM
    // for (int i = 1; i < cluster.num_blocks() - 1; i++) {
    //     if (tid == 0) {
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
    //         uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_q));
    //         uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
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
    //     if (tid < HEAD_DIM / 2) {
    //         half2 buffer = *(half2*)(&local_buffer[tid * 2]);
    //         if (i == cluster.num_blocks() - 2) // ReLU
    //             *(half2*)(&local_q[tid * 2]) = __hmax2(__hadd2(*(half2*)(&local_q[tid * 2]), buffer), __float22half2_rn({0.0f, 0.0f}));
    //         else
    //             *(half2*)(&local_q[tid * 2]) = __hadd2(*(half2*)(&local_q[tid * 2]), buffer);
    //     }
    //     cluster.sync();
    // }

    // // Compute up proj
    // __shared__ __align__(16) half local_q_up[HEAD_DIM];
    // tmp = 0.0;
    // for (int i = 0; i < FFN_DIM_PER_BLOCK; i+=NUM_PER_ROW) { // 16
    //     *(uint4*)(&reg_input[0][0]) = *(uint4*)(&input_shmem[input_idx + i]);
    //     *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&ffn_up[weight_idx + i]);
    //     #pragma unroll
    //     for (int d = 0; d < NUM_PER_THREAD; d++) {
    //         tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
    //     }
    // }
    // #pragma unroll
    // for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
    //     tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    // }
    // if (lane_id % NUM_THREAD_PER_ROW == 0) {
    //     local_q_up[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    // }
    // block.sync();

    // // up proj reduce through DSM
    // for (int i = 1; i < cluster.num_blocks() - 1; i++) {
    //     if (tid == 0) {
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
    //         uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_q_up));
    //         uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
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
    //     if (tid < HEAD_DIM / 2) {
    //         half2 buffer = *(half2*)(&local_buffer[tid * 2]);
    //         *(half2*)(&local_q_up[tid * 2]) = __hadd2(*(half2*)(&local_q_up[tid * 2]), buffer);
    //     }
    //     cluster.sync();
    // }

    // // Compute up_gate mul and down_proj
    // half __align__(16) reg_input_2[1][NUM_PER_THREAD];
    // weight_idx = head_id * HEAD_DIM + cluster_block_id * FFN_DIM_PER_BLOCK * HIDDEN_DIM + warp_id * NUM_ROW_PER_WARP_4 * HIDDEN_DIM + (lane_id / NUM_THREAD_PER_ROW) * HIDDEN_DIM + (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    // for (int i = 0; i < NUM_ROW_PER_WARP_4; i+=NUM_ROW_PER_WARP) {
    //     tmp = 0.0;
    //     for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW) {
    //         *(uint4*)(&reg_input[0][0]) = *(uint4*)(&local_q[input_idx + j]);
    //         *(uint4*)(&reg_input_2[0][0]) = *(uint4*)(&local_q_up[input_idx + j]);
    //         *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&ffn_down[weight_idx + i * HIDDEN_DIM + j]);
    //         #pragma unroll
    //         for (int d = 0; d < NUM_PER_THREAD; d++) {
    //             tmp += __half2float(reg_input[0][d] * reg_input_2[0][d] * reg_weight[0][d]);
    //         }
    //     }
    //     #pragma unroll
    //     for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
    //         tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    //     }
    //     if (lane_id % NUM_THREAD_PER_ROW == 0) {
    //         atomicAdd(&output[cluster_block_id * DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP_2 + i + lane_id / NUM_THREAD_PER_ROW], __float2half(tmp));
    //     }
    // }
}

int main(int argc, char** argv) {
    cudaFuncSetAttribute(single_decode, cudaFuncAttributeNonPortableClusterSizeAllowed, 16);

    half *h_input, *d_input;
    half *h_kv_cache, *d_kv_cache;
    half *h_w_qkv, *d_w_qkv;
    half *h_w_o, *d_w_o;
    half *h_ffn_gate, *d_ffn_gate;
    half *h_ffn_down, *d_ffn_down;
    half *h_ffn_up, *d_ffn_up;
    half *h_rms_input, *d_rms_input;
    half *h_rms_attn, *d_rms_attn;
    float *h_cos, *d_cos;
    float *h_sin, *d_sin;
    h_input = new half[1 * HIDDEN_DIM];
    h_w_qkv = new half[3 * HIDDEN_DIM * HIDDEN_DIM];
    h_w_o = new half[HIDDEN_DIM * HIDDEN_DIM];
    h_kv_cache = new half[2 * SEQ_LEN * HEAD_NUM * HEAD_DIM];
    h_ffn_gate = new half[HIDDEN_DIM * FFN_DIM];
    h_ffn_up = new half[HIDDEN_DIM * FFN_DIM];
    h_ffn_down = new half[FFN_DIM * HIDDEN_DIM];
    h_rms_input = new half[HIDDEN_DIM];
    h_rms_attn = new half[HIDDEN_DIM];
    h_cos = new float[HEAD_DIM];
    h_sin = new float[HEAD_DIM];

    fill_matrix(h_input, 1 * HIDDEN_DIM);
    fill_matrix(h_w_qkv, 3 * HIDDEN_DIM * HIDDEN_DIM);
    fill_matrix(h_w_o, HIDDEN_DIM * HIDDEN_DIM);
    fill_matrix(h_kv_cache, 2 * SEQ_LEN * HEAD_NUM * HEAD_DIM);
    fill_matrix(h_ffn_gate, HIDDEN_DIM * FFN_DIM);
    fill_matrix(h_ffn_down, FFN_DIM * HIDDEN_DIM);
    fill_matrix(h_ffn_up, HIDDEN_DIM * FFN_DIM);
    fill_matrix(h_rms_input, HIDDEN_DIM);
    fill_matrix(h_rms_attn, HIDDEN_DIM);

    // Init cos, sin used in RoPE
    int encode_point_offset = 0;
    float rope_scale = 1;
    float rope_theta = 500000;
    for (int j = 0; j < HEAD_DIM; j++) {
        float inv_freq =(encode_point_offset / rope_scale) / (std::pow(rope_theta, float(2 * (j % (HEAD_DIM / 2))) / float(HEAD_DIM)));
        h_cos[j] = std::cos(inv_freq);
        h_sin[j] = std::sin(inv_freq);
    }

    cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * 1 * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_qkv), sizeof(half) * 3 * HIDDEN_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_o), sizeof(half) * HIDDEN_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_kv_cache), sizeof(half) * 2 * SEQ_LEN * HEAD_NUM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_gate), sizeof(half) * HIDDEN_DIM * FFN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_down), sizeof(half) * FFN_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_up), sizeof(half) * HIDDEN_DIM * FFN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_input), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_attn), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_cos), sizeof(float) * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_sin), sizeof(float) * HEAD_DIM);

    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_qkv), h_w_qkv, sizeof(half) * 3 * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_o), h_w_o, sizeof(half) * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_kv_cache), h_kv_cache, sizeof(half) * 2 * SEQ_LEN * HEAD_NUM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_gate), h_ffn_gate, sizeof(half) * HIDDEN_DIM * FFN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_down), h_ffn_down, sizeof(half) * FFN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_up), h_ffn_up, sizeof(half) * HIDDEN_DIM * FFN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_input), h_rms_input, sizeof(half) * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_attn), h_rms_attn, sizeof(half) * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_cos), h_cos, sizeof(float) * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_sin), h_sin, sizeof(float) * HEAD_DIM, cudaMemcpyHostToDevice);

    half* h_output, *d_output;
    h_output = new half[1 * HIDDEN_DIM];
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * 1 * HIDDEN_DIM);
    
    half *global_reduce;
    cudaMalloc(reinterpret_cast<void**>(&global_reduce), sizeof(half) * HIDDEN_DIM);

    CUtensorMap tensor_map_weight{};
    CUtensorMap tensor_map_kv_cache{};
    CUtensorMap tensor_map_weight_o{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {HIDDEN_DIM, 3 * HIDDEN_DIM};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map_weight,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_w_qkv,                 // void *globalAddress,
        size,                       // const cuuint64_t *globalDim,
        stride,                     // const cuuint64_t *globalStrides,
        box_size,                   // const cuuint32_t *boxDim,
        elem_stride,                // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_kv_cache[rank] = {HIDDEN_DIM, 2 * SEQ_LEN};
    uint64_t stride_kv_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_kv_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    uint32_t elem_stride_kv_cache[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_kv_cache = cuTensorMapEncodeTiled(
        &tensor_map_kv_cache,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_kv_cache,                 // void *globalAddress,
        size_kv_cache,                       // const cuuint64_t *globalDim,
        stride_kv_cache,                     // const cuuint64_t *globalStrides,
        box_size_kv_cache,                   // const cuuint32_t *boxDim,
        elem_stride_kv_cache,                // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_weight_o[rank] = {HIDDEN_DIM, HIDDEN_DIM};
    uint64_t stride_weight_o[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_o[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_weight_o[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_weight_o = cuTensorMapEncodeTiled(
        &tensor_map_weight_o,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_w_o,                 // void *globalAddress,
        size_weight_o,                       // const cuuint64_t *globalDim,
        stride_weight_o,                     // const cuuint64_t *globalStrides,
        box_size_weight_o,                   // const cuuint32_t *boxDim,
        elem_stride_weight_o,                // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    int wmup = 100;
    int test = 100;
    for (int i = 0; i < wmup; i++) {
        single_decode<<<grid, block>>>(
            d_output,
            d_input,
            d_w_qkv,
            d_kv_cache,
            d_ffn_gate,
            d_ffn_down,
            d_ffn_up,
            global_reduce,
            d_rms_input,
            d_rms_attn,
            d_cos,
            d_sin,
            tensor_map_weight,
            tensor_map_kv_cache,
            tensor_map_weight_o
        );
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    cudaEventRecord(st);
    for (int i = 0; i < test; i++) {
        single_decode<<<grid, block>>>(
            d_output,
            d_input,
            d_w_qkv,
            d_kv_cache,
            d_ffn_gate,
            d_ffn_down,
            d_ffn_up,
            global_reduce,
            d_rms_input,
            d_rms_attn,
            d_cos,
            d_sin,
            tensor_map_weight,
            tensor_map_kv_cache,
            tensor_map_weight_o
        );
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / test * 1e3 << " us" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyDeviceToHost);
    return 0;
}