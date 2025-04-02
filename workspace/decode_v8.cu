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

// nvcc --generate-code=arch=compute_90a,code=sm_90a -O3 -std=c++17 -lcuda decode_v8.cu -o test && ./test

#define HEAD_DIM 128    
#define HEAD_NUM 32     
#define FFN_DIM 12288   
#define HIDDEN_DIM 4096 
#define SEQ_LEN 16384

#define NUM_WARPS 4 // 4 8 16 32
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE) 
#define CLUSTER_SIZE 4 // 2 4
#define NUM_PER_THREAD 8
#define NUM_ROW_PER_WARP (HEAD_DIM / NUM_WARPS) 
#define NUM_THREAD_PER_ROW (WARP_SIZE / NUM_ROW_PER_WARP) 
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW) 
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)
#define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) 
#define FFN_DIM_PER_CLUSTER (FFN_DIM / HEAD_NUM) 
#define MAX_SMEM_DIM FFN_DIM_PER_CLUSTER

#define TMA_LOAD_ONCE 64 // 8 16 32 64 128 256
#define TMA_LOAD_ONCE_MAX 256
#define TMA_LOAD_ONCE_NUM (TMA_LOAD_ONCE * HEAD_DIM)
#define TMA_LOAD_ONCE_SIZE (TMA_LOAD_ONCE_NUM * sizeof(half))
#define TMA_LOAD_ONCE_ATTN (TMA_LOAD_ONCE / 2)
#define TMA_LOAD_ONCE_NUM_ATTN ((TMA_LOAD_ONCE * HEAD_DIM) / 2)
#define TMA_LOAD_ONCE_SIZE_ATTN (TMA_LOAD_ONCE_NUM_ATTN * sizeof(half))
#define TMA_LOAD_ONCE_NUM_FFN (TMA_LOAD_ONCE * TMA_LOAD_ONCE_MAX)
#define TMA_LOAD_ONCE_NUM_FFN_TOTAL (TMA_LOAD_ONCE * FFN_DIM_PER_CLUSTER)
#define TMA_LOAD_ONCE_SIZE_FFN (TMA_LOAD_ONCE_NUM_FFN_TOTAL * sizeof(half))

#define NUM_THREAD_PER_ROW_2 (HEAD_DIM / NUM_PER_THREAD) // 16
#define NUM_ROW_PER_WARP_2 (WARP_SIZE / NUM_THREAD_PER_ROW_2) // 2
#define NUM_PER_ROW_2 (NUM_WARPS * NUM_ROW_PER_WARP_2) // 8
#define DEC_TILE (TMA_LOAD_ONCE_ATTN / NUM_PER_ROW_2)
#define NUM_ROW_PER_WARP_3 (TMA_LOAD_ONCE / NUM_WARPS) 
#define NUM_THREAD_PER_ROW_3 (WARP_SIZE / NUM_ROW_PER_WARP_3) 
#define NUM_PER_ROW_3 (NUM_PER_THREAD * NUM_THREAD_PER_ROW_3) 

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
    half* global_reduce,    // hidden_dim  
    half* w_rms_input,// hidden_dim
    half* w_rms_attn, // hidden_dim
    float* cos,       // head_dim
    float* sin,       // head_dim
    const __grid_constant__ CUtensorMap tensor_map, // 3 * hidden_dim * hidden_dim
    const __grid_constant__ CUtensorMap tensor_map_k_cache, // seqlen * head_num * head_dim
    const __grid_constant__ CUtensorMap tensor_map_v_cache, // seqlen * head_num * head_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_o, // hidden_dim * hidden_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_gate_up, // 2 * hidden_dim * ffn_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_gate_up_,// 2 * hidden_dim * ffn_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_down, // ffn_dim * hidden_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_down_ // ffn_dim * hidden_dim
)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank() % HEAD_NUM;
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; 
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t tile_row = tid / NUM_THREAD_PER_ROW_2;
    const uint32_t tile_col = tid % NUM_THREAD_PER_ROW_2;

    // Init shared memory
    __shared__ __align__(16) half input_shmem[DIM_PER_BLOCK];
    __shared__ float reduction[NUM_WARPS];
    __shared__ float cluster_local_sum;
    __shared__ alignas(128) half weight[2 * TMA_LOAD_ONCE * MAX_SMEM_DIM];
    __shared__ __align__(16) half local_qkv[MAX_SMEM_DIM + MAX_SMEM_DIM + HEAD_DIM];
    __shared__ __align__(16) half local_output[HEAD_DIM];

    // Init register
    float local_sum = 0;
    half __align__(16) reg_input_norm[2], reg_weight_norm[2];
    float tmp = 0.0;
    half __align__(16) reg_input[NUM_PER_THREAD];
    half __align__(16) reg_weight[NUM_PER_THREAD];
    half2 q_rope, q_rope_1;
    half2 k_rope, k_rope_1;
    float2 cos_reg, sin_reg;
    uint32_t size;
    half2 buffer;
    half __align__(16) reg_reduce[NUM_PER_THREAD];
    float __align__(16) qk[DEC_TILE];
    float tmp_ffn[FFN_DIM_PER_CLUSTER / HEAD_DIM];
    for (int j = 0; j < FFN_DIM_PER_CLUSTER / HEAD_DIM; j++){
      tmp_ffn[j] = 0.0;
    }

    // Init barrier
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[4];
    barrier::arrival_token token[4];
    __shared__ uint64_t barrier;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    if (tid == 0) {
        init(&bar[0], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[1], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[2], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[3], blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    block.sync();

    // Precompute some indices
    uint cluster_block_st_id = cluster_block_id * DIM_PER_BLOCK;
    uint input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    uint input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    uint weight_idx_2 = warp_id * NUM_PER_ROW_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
    uint input_idx_3 = (lane_id % NUM_THREAD_PER_ROW_3) * NUM_PER_THREAD;
    uint weight_idx_3 = warp_id * NUM_ROW_PER_WARP_3 + lane_id / NUM_THREAD_PER_ROW_3;

    // Load input to shared memory
    #pragma unroll
    for (int i = tid * 8; i < DIM_PER_BLOCK; i+=BLOCK_SIZE * 8) {
        *(uint4*)(&input_shmem[i]) = *(uint4*)(&input[cluster_block_st_id + i]);
    }
    block.sync();

    // RMSNorm
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
    float eps = 1e-6;
    half rms_rcp = __float2half(1.f / (std::sqrt(cluster_local_sum / float(HIDDEN_DIM)) + eps));
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
        *(half2*)(&reg_input_norm[0]) = __hmul2(*(half2*)(&reg_input_norm[0]), {rms_rcp, rms_rcp});
        *(half2*)(&reg_weight_norm[0]) = *(half2*)(&w_rms_input[cluster_block_st_id + d]);
        *(half2*)(&input_shmem[d]) = __hmul2(*(half2*)(&reg_input_norm[0]), *(half2*)(&reg_weight_norm[0]));
    }
    block.sync();

    // Compute input @ w_q
    // Preload weight_q
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, head_id * HEAD_DIM, cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, head_id * HEAD_DIM, cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d] * weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }

    // Compute input @ w_k
    // Preload weight_k
    tmp = 0.0;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d] * weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[HEAD_DIM + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    
    // Compute input @ w_v
    // Preload weight_v
    tmp = 0.0;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM * 2 + cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM * 2 + cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d] * weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[HEAD_DIM * 2 + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }

    // DSM Ring-All reduce
    size = (HEAD_DIM * 3) * sizeof(half);
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
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
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

        // Local reduce-add
        if (tid < HEAD_DIM / 2) {
            for (int j = 0; j < 3; j++) {
                buffer = *(half2*)(&weight[j * HEAD_DIM + tid * 2]);
                *(half2*)(&local_qkv[j * HEAD_DIM + tid * 2]) = __hadd2(*(half2*)(&local_qkv[j * HEAD_DIM + tid * 2]), buffer);
            }
        }
        cluster.sync();
    }
    // if(head_id == 0 && cluster_block_id == 0 && tid == 0)
    //     printf("%f, %f, %f, %f, %f, %f \n", __half2float(local_qkv[0]), __half2float(local_qkv[127]), __half2float(local_qkv[HEAD_DIM + 0]), __half2float(local_qkv[HEAD_DIM + 127]), __half2float(local_qkv[HEAD_DIM * 2 + 0]), __half2float(local_qkv[HEAD_DIM * 2 + 127]));

    // Compute RoPE
    if (tid < HEAD_DIM / 2) {
        q_rope = *(half2*)(&local_qkv[tid * 2]);
        k_rope = *(half2*)(&local_qkv[HEAD_DIM + tid * 2]);
        if (tid * 2 < HEAD_DIM / 2) {
            q_rope_1 = *(half2*)(&local_qkv[HEAD_DIM / 2 + tid * 2]);
            k_rope_1 = *(half2*)(&local_qkv[HEAD_DIM + HEAD_DIM / 2 + tid * 2]);
            cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
            sin_reg = {-sin[HEAD_DIM / 2 + tid * 2], -sin[HEAD_DIM / 2 + tid * 2 + 1]};
        } else {
            q_rope_1 = *(half2*)(&local_qkv[tid * 2 - HEAD_DIM / 2]);
            k_rope_1 = *(half2*)(&local_qkv[HEAD_DIM + tid * 2 - HEAD_DIM / 2]);
            cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
            sin_reg = {sin[tid * 2 - HEAD_DIM / 2], sin[tid * 2 + 1 - HEAD_DIM / 2]};
        }
        *(half2*)(&local_qkv[tid * 2]) = __hadd2(__hmul2(q_rope, __float22half2_rn(cos_reg)), __hmul2(q_rope_1, __float22half2_rn(sin_reg)));
        *(half2*)(&local_qkv[HEAD_DIM + tid * 2]) = __hadd2(__hmul2(k_rope, __float22half2_rn(cos_reg)), __hmul2(k_rope_1, __float22half2_rn(sin_reg)));
    }

    // Compute flash-decoding
    local_sum = 0.0f;
    if(lane_id == 0)
        reduction[warp_id] = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = __float2half(0.0f);
    *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2]);
    block.sync();

    // Preload kv_cache
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_k_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_v_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK, bar[2]);
        token[2] = cuda::device::barrier_arrive_tx(bar[2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
    } else {
        token[0] = bar[0].arrive();
        token[2] = bar[2].arrive();
    }

    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_k_cache, head_id * HEAD_DIM, cluster_block_st_id + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            qk[j] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                qk[j] += __half2float(reg_input[d] * reg_weight[d]);
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                qk[j] += __shfl_down_sync(0xffffffff, qk[j], mask);
            }
            qk[j] = __expf(qk[j] * __frsqrt_rn(HEAD_DIM));
            local_sum += qk[j];
        }

        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_v_cache, head_id * HEAD_DIM, cluster_block_st_id + id * TMA_LOAD_ONCE_ATTN, bar[2 + id % 2]);
            token[2 + id % 2] = cuda::device::barrier_arrive_tx(bar[2 + id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[2 + id % 2] = bar[2 + id % 2].arrive();
        }
        bar[2 + (id - 1) % 2].wait(std::move(token[2 + (id - 1) % 2]));
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                reg_reduce[d] += __float2half(qk[j] * __half2float(reg_weight[d]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int j = 0; j < DEC_TILE; j++) {
        if (cluster_block_id == CLUSTER_SIZE - 1 && warp_id == NUM_WARPS - 1 && lane_id / NUM_THREAD_PER_ROW_2 == 1 && j == DEC_TILE - 1)
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[HEAD_DIM + input_idx_2]);
        else
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        qk[j] = 0.0f;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            qk[j] += __half2float(reg_input[d] * reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[j] += __shfl_down_sync(0xffffffff, qk[j], mask);
        }
        qk[j] = __expf(qk[j] * __frsqrt_rn(HEAD_DIM));
        local_sum += qk[j];
    }
    bar[3].wait(std::move(token[3]));
    for (int j = 0; j < DEC_TILE; j++) {
        if (cluster_block_id == CLUSTER_SIZE - 1 && warp_id == NUM_WARPS - 1 && lane_id / NUM_THREAD_PER_ROW_2 == 1 && j == DEC_TILE - 1) 
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_2]);
        else
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] += __float2half(qk[j] * __half2float(reg_weight[d]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        atomicAdd(&reduction[warp_id], local_sum);
    }
    *(uint4*)(&weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = __float2half(0.0f);
    block.sync();
    if (tid < NUM_WARPS) {
        local_sum = reduction[tid];
    }
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }
    if(tid == 0) {
        cluster_local_sum = local_sum;
    }
    cluster.sync();
    // DSM Ring-All reduce
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            float* dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
            atomicAdd(dst_shmem, local_sum);
        }
        cluster.sync();
    }
    for (int i = 0; i < NUM_PER_ROW_2; i++) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[i * HEAD_DIM + tile_col * NUM_PER_THREAD]);
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++)
            reg_reduce[j] += reg_input[j];
    }
    if(tid < NUM_THREAD_PER_ROW_2) {
        *(uint4*)(&local_output[tid * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++)
            local_output[tid * NUM_PER_THREAD + j] = __float2half(__half2float(local_output[tid * NUM_PER_THREAD + j]) * __frcp_rn(cluster_local_sum));
    }
    block.sync();

    // DSM Ring-All reduce
    size = HEAD_DIM * sizeof(half);
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
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_output));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
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
            buffer = *(half2*)(&weight[tid * 2]);
            *(half2*)(&local_output[tid * 2]) = __hadd2(*(half2*)(&local_output[tid * 2]), buffer);
        }
        cluster.sync();
    }
    // if(head_id == 0 && cluster_block_id == 1 && tid == 0)
    //     printf("%f, %f \n", __half2float(local_output[0]), __half2float(local_output[127]));
    // Compute output @ w_o
    // Preload w_o
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_o, cluster_block_st_id, head_id * HEAD_DIM, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_o, cluster_block_st_id + id * TMA_LOAD_ONCE, head_id * HEAD_DIM, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        tmp = 0.0;
        for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_3) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_output[input_idx_3 + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d] * weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
        if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
            atomicAdd(&global_reduce[cluster_block_st_id + weight_idx_3 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
        }
    }
    bar[1].wait(std::move(token[1]));
    tmp = 0.0;
    for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_3) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_output[input_idx_3 + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * weight[TMA_LOAD_ONCE_NUM + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
        atomicAdd(&global_reduce[cluster_block_st_id + weight_idx_3 + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    }
    cluster.sync();

    // Fused residual and RMSNorm
    local_sum = 0.0;
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        // printf("%f, %f \n", __half2float(input_shmem[d]), __half2float(global_reduce[cluster_block_st_id + d]));
        *(half2*)(&reg_input_norm[0]) = __hadd2(*(half2*)(&input_shmem[d]), *(half2*)(&global_reduce[cluster_block_st_id + d]));
        *(half2*)(&input_shmem[d]) = *(half2*)(&reg_input_norm[0]);
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
    if (tid < NUM_WARPS){
        local_sum = reduction[tid];
    }
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
    rms_rcp = __float2half(1.f / (std::sqrt(cluster_local_sum / float(HIDDEN_DIM)) + eps));
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
        *(half2*)(&reg_input_norm[0]) = __hmul2(*(half2*)(&reg_input_norm[0]), {rms_rcp, rms_rcp});
        *(half2*)(&reg_weight_norm[0]) = *(half2*)(&w_rms_attn[cluster_block_st_id + d]);
        *(half2*)(&input_shmem[d]) = __hmul2(*(half2*)(&reg_input_norm[0]), *(half2*)(&reg_weight_norm[0]));
    }
    block.sync();

    // Compute input @ ffn_gate
    // Preload weight_gate
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_gate_up, head_id * FFN_DIM_PER_CLUSTER, cluster_block_st_id, bar[0]);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_gate_up_, head_id * FFN_DIM_PER_CLUSTER + TMA_LOAD_ONCE_MAX, cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_FFN);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL], &tensor_map_weight_gate_up, head_id * FFN_DIM_PER_CLUSTER, cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_gate_up_, head_id * FFN_DIM_PER_CLUSTER + TMA_LOAD_ONCE_MAX, cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_FFN);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            for (int j = 0; j < TMA_LOAD_ONCE_MAX / HEAD_DIM; j++) {
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp_ffn[j] += __half2float(reg_input[d] * weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx + i + d) * TMA_LOAD_ONCE_MAX + weight_idx + j * HEAD_DIM]);
                }
            }
            for (int j = 0; j < (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) / HEAD_DIM; j++) {
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp_ffn[TMA_LOAD_ONCE_MAX / HEAD_DIM + j] += __half2float(reg_input[d] * weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx + i + d) * (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) + weight_idx + j * HEAD_DIM]);
                }
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        for (int j = 0; j < TMA_LOAD_ONCE_MAX / HEAD_DIM; j++) {
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp_ffn[j] += __half2float(reg_input[d] * weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx + i + d) * TMA_LOAD_ONCE_MAX + weight_idx + j * HEAD_DIM]);
            }
        }
        for (int j = 0; j < (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) / HEAD_DIM; j++) {
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp_ffn[TMA_LOAD_ONCE_MAX / HEAD_DIM + j] += __half2float(reg_input[d] * weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx + i + d) * (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) + weight_idx + j * HEAD_DIM]);
            }
        }
    }
    for (int j = 0; j < FFN_DIM_PER_CLUSTER / HEAD_DIM; j++){
        local_qkv[j * HEAD_DIM + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp_ffn[j]);
    }

    // Compute input @ ffn_up
    for (int j = 0; j < FFN_DIM_PER_CLUSTER / HEAD_DIM; j++){
        tmp_ffn[j] = 0.0;
    }
    // Preload weight_up
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_gate_up, head_id * FFN_DIM_PER_CLUSTER, HIDDEN_DIM + cluster_block_st_id, bar[0]);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_gate_up_, head_id * FFN_DIM_PER_CLUSTER + TMA_LOAD_ONCE_MAX, HIDDEN_DIM + cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_FFN);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL], &tensor_map_weight_gate_up, head_id * FFN_DIM_PER_CLUSTER, HIDDEN_DIM + cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_gate_up_, head_id * FFN_DIM_PER_CLUSTER + TMA_LOAD_ONCE_MAX, HIDDEN_DIM + cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_FFN);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            for (int j = 0; j < TMA_LOAD_ONCE_MAX / HEAD_DIM; j++) {
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp_ffn[j] += __half2float(reg_input[d] * weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx + i + d) * TMA_LOAD_ONCE_MAX + weight_idx + j * HEAD_DIM]);
                }
            }
            for (int j = 0; j < (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) / HEAD_DIM; j++) {
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp_ffn[TMA_LOAD_ONCE_MAX / HEAD_DIM + j] += __half2float(reg_input[d] * weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx + i + d) * (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) + weight_idx + j * HEAD_DIM]);
                }
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        for (int j = 0; j < TMA_LOAD_ONCE_MAX / HEAD_DIM; j++) {
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp_ffn[j] += __half2float(reg_input[d] * weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx + i + d) * TMA_LOAD_ONCE_MAX + weight_idx + j * HEAD_DIM]);
            }
        }
        for (int j = 0; j < (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) / HEAD_DIM; j++) {
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp_ffn[TMA_LOAD_ONCE_MAX / HEAD_DIM + j] += __half2float(reg_input[d] * weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx + i + d) * (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) + weight_idx + j * HEAD_DIM]);
            }
        }
    }
    for (int j = 0; j < FFN_DIM_PER_CLUSTER / HEAD_DIM; j++){
        local_qkv[MAX_SMEM_DIM + j * HEAD_DIM + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp_ffn[j]);
    }
    block.sync();

    // DSM All-reduce
    size = FFN_DIM_PER_CLUSTER * 2 * sizeof(half);
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
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
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
            for (int j = 0; j < FFN_DIM_PER_CLUSTER / HEAD_DIM; j++) {
              buffer = *(half2*)(&weight[j * HEAD_DIM + tid * 2]);
              if (i == cluster.num_blocks() - 2) // ReLU
                  *(half2*)(&local_qkv[j * HEAD_DIM + tid * 2]) = __hmax2(__hadd2(*(half2*)(&local_qkv[j * HEAD_DIM + tid * 2]), buffer), __float22half2_rn({0.0f, 0.0f}));
              else
                  *(half2*)(&local_qkv[j * HEAD_DIM + tid * 2]) = __hadd2(*(half2*)(&local_qkv[j * HEAD_DIM + tid * 2]), buffer);
            }
            for (int j = 0; j < FFN_DIM_PER_CLUSTER / HEAD_DIM; j++) {
                buffer = *(half2*)(&weight[FFN_DIM_PER_CLUSTER + j * HEAD_DIM + tid * 2]);
                *(half2*)(&local_qkv[FFN_DIM_PER_CLUSTER + j * HEAD_DIM + tid * 2]) = __hadd2(*(half2*)(&local_qkv[FFN_DIM_PER_CLUSTER + j * HEAD_DIM + tid * 2]), buffer);
            }
        }
        cluster.sync();
    }
    
    // Compute up_gate mul and down_proj
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_down, cluster_block_st_id, head_id * FFN_DIM_PER_CLUSTER, bar[0]);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_down_, cluster_block_st_id, head_id * FFN_DIM_PER_CLUSTER + TMA_LOAD_ONCE_MAX, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_FFN);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL], &tensor_map_weight_down, cluster_block_st_id + id * TMA_LOAD_ONCE, head_id * FFN_DIM_PER_CLUSTER, bar[id % 2]);
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_down_, cluster_block_st_id + id * TMA_LOAD_ONCE, head_id * FFN_DIM_PER_CLUSTER + TMA_LOAD_ONCE_MAX, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_FFN);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        tmp = 0.0;
        for (int j = 0; j < TMA_LOAD_ONCE_MAX; j+=NUM_PER_ROW_3) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3 + j]);
            *(uint4*)(&reg_reduce[0]) = *(uint4*)(&local_qkv[MAX_SMEM_DIM + input_idx_3 + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d] * reg_reduce[d] * weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]);
            }
        }
        for (int j = 0; j < FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX; j+=NUM_PER_ROW_3) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3 + TMA_LOAD_ONCE_MAX + j]);
            *(uint4*)(&reg_reduce[0]) = *(uint4*)(&local_qkv[MAX_SMEM_DIM + input_idx_3 + TMA_LOAD_ONCE_MAX + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d] * reg_reduce[d] * weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
        if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
            atomicAdd(&output[cluster_block_st_id + weight_idx_3 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
        }
    }
    bar[1].wait(std::move(token[1]));
    tmp = 0.0;
    for (int j = 0; j < TMA_LOAD_ONCE_MAX; j+=NUM_PER_ROW_3) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3 + j]);
        *(uint4*)(&reg_reduce[0]) = *(uint4*)(&local_qkv[MAX_SMEM_DIM + input_idx_3 + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * reg_reduce[d] * weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]);
        }
    }
    for (int j = 0; j < FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX; j+=NUM_PER_ROW_3) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3 + TMA_LOAD_ONCE_MAX + j]);
        *(uint4*)(&reg_reduce[0]) = *(uint4*)(&local_qkv[MAX_SMEM_DIM + input_idx_3 + TMA_LOAD_ONCE_MAX + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d] * reg_reduce[d] * weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
        atomicAdd(&output[cluster_block_st_id + weight_idx_3 + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    }
}

int main(int argc, char** argv) {
    cudaFuncSetAttribute(single_decode, cudaFuncAttributeNonPortableClusterSizeAllowed, 16);
    uint32_t max_shmem_size = 0;
    cudaFuncSetAttribute(single_decode, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    half *h_input, *d_input;
    half *h_k_cache, *d_k_cache;
    half *h_v_cache, *d_v_cache;
    half *h_w_qkv, *d_w_qkv;
    half *h_w_o, *d_w_o;
    half *h_ffn_gate_up, *d_ffn_gate_up;
    half *h_ffn_down, *d_ffn_down;
    half *h_rms_input, *d_rms_input;
    half *h_rms_attn, *d_rms_attn;
    float *h_cos, *d_cos;
    float *h_sin, *d_sin;
    h_input = new half[1 * HIDDEN_DIM];
    h_w_qkv = new half[3 * HIDDEN_DIM * HIDDEN_DIM];
    h_w_o = new half[HIDDEN_DIM * HIDDEN_DIM];
    h_k_cache = new half[SEQ_LEN * HEAD_NUM * HEAD_DIM];
    h_v_cache = new half[SEQ_LEN * HEAD_NUM * HEAD_DIM];
    h_ffn_gate_up = new half[2 * HIDDEN_DIM * FFN_DIM];
    h_ffn_down = new half[FFN_DIM * HIDDEN_DIM];
    h_rms_input = new half[HIDDEN_DIM];
    h_rms_attn = new half[HIDDEN_DIM];
    h_cos = new float[HEAD_DIM];
    h_sin = new float[HEAD_DIM];

    fill_matrix(h_input, 1 * HIDDEN_DIM);
    fill_matrix(h_w_qkv, 3 * HIDDEN_DIM * HIDDEN_DIM);
    fill_matrix(h_w_o, HIDDEN_DIM * HIDDEN_DIM);
    fill_matrix(h_k_cache, SEQ_LEN * HEAD_NUM * HEAD_DIM);
    fill_matrix(h_v_cache, SEQ_LEN * HEAD_NUM * HEAD_DIM);
    fill_matrix(h_ffn_gate_up, 2 * HIDDEN_DIM * FFN_DIM);
    fill_matrix(h_ffn_down, FFN_DIM * HIDDEN_DIM);
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
    cudaMalloc(reinterpret_cast<void**>(&d_k_cache), sizeof(half) * SEQ_LEN * HEAD_NUM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_v_cache), sizeof(half) * SEQ_LEN * HEAD_NUM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_gate_up), sizeof(half) * 2 * HIDDEN_DIM * FFN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_down), sizeof(half) * FFN_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_input), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_attn), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_cos), sizeof(float) * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_sin), sizeof(float) * HEAD_DIM);

    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_qkv), h_w_qkv, sizeof(half) * 3 * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_o), h_w_o, sizeof(half) * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_k_cache), h_k_cache, sizeof(half) * SEQ_LEN * HEAD_NUM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_v_cache), h_v_cache, sizeof(half) * SEQ_LEN * HEAD_NUM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_gate_up), h_ffn_gate_up, sizeof(half) * 2 * HIDDEN_DIM * FFN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_down), h_ffn_down, sizeof(half) * FFN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
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
    CUtensorMap tensor_map_k_cache{};
    CUtensorMap tensor_map_v_cache{};
    CUtensorMap tensor_map_weight_o{};
    CUtensorMap tensor_map_weight_gate_up{};
    CUtensorMap tensor_map_weight_gate_up_{};
    CUtensorMap tensor_map_weight_down{};
    CUtensorMap tensor_map_weight_down_{};
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

    uint64_t size_k_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_k_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_k_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_k_cache[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_k_cache = cuTensorMapEncodeTiled(
        &tensor_map_k_cache,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_k_cache,                 // void *globalAddress,
        size_k_cache,                       // const cuuint64_t *globalDim,
        stride_k_cache,                     // const cuuint64_t *globalStrides,
        box_size_k_cache,                   // const cuuint32_t *boxDim,
        elem_stride_k_cache,                // const cuuint32_t *elementStrides,
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

    uint64_t size_v_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_v_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_v_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_v_cache[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_v_cache = cuTensorMapEncodeTiled(
        &tensor_map_v_cache,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_v_cache,                 // void *globalAddress,
        size_v_cache,                       // const cuuint64_t *globalDim,
        stride_v_cache,                     // const cuuint64_t *globalStrides,
        box_size_v_cache,                   // const cuuint32_t *boxDim,
        elem_stride_v_cache,                // const cuuint32_t *elementStrides,
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

    uint64_t size_weight_gate_up[rank] = {FFN_DIM, 2 * HIDDEN_DIM};
    uint64_t stride_weight_gate_up[rank - 1] = {FFN_DIM * sizeof(half)};
    uint32_t box_size_weight_gate_up[rank] = {TMA_LOAD_ONCE_MAX, TMA_LOAD_ONCE};
    uint32_t elem_stride_weight_gate_up[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_weight_gate_up = cuTensorMapEncodeTiled(
        &tensor_map_weight_gate_up,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_ffn_gate_up,                 // void *globalAddress,
        size_weight_gate_up,                       // const cuuint64_t *globalDim,
        stride_weight_gate_up,                     // const cuuint64_t *globalStrides,
        box_size_weight_gate_up,                   // const cuuint32_t *boxDim,
        elem_stride_weight_gate_up,                // const cuuint32_t *elementStrides,
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

    uint64_t size_weight_gate_up_[rank] = {FFN_DIM, 2 * HIDDEN_DIM};
    uint64_t stride_weight_gate_up_[rank - 1] = {FFN_DIM * sizeof(half)};
    uint32_t box_size_weight_gate_up_[rank] = {FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX, TMA_LOAD_ONCE};
    uint32_t elem_stride_weight_gate_up_[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_weight_gate_up_ = cuTensorMapEncodeTiled(
        &tensor_map_weight_gate_up_,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_ffn_gate_up,                 // void *globalAddress,
        size_weight_gate_up_,                       // const cuuint64_t *globalDim,
        stride_weight_gate_up_,                     // const cuuint64_t *globalStrides,
        box_size_weight_gate_up_,                   // const cuuint32_t *boxDim,
        elem_stride_weight_gate_up_,                // const cuuint32_t *elementStrides,
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

    uint64_t size_weight_down[rank] = {HIDDEN_DIM, FFN_DIM};
    uint64_t stride_weight_down[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_down[rank] = {TMA_LOAD_ONCE, TMA_LOAD_ONCE_MAX};
    uint32_t elem_stride_weight_down[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_weight_down = cuTensorMapEncodeTiled(
        &tensor_map_weight_down,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_ffn_down,                 // void *globalAddress,
        size_weight_down,                       // const cuuint64_t *globalDim,
        stride_weight_down,                     // const cuuint64_t *globalStrides,
        box_size_weight_down,                   // const cuuint32_t *boxDim,
        elem_stride_weight_down,                // const cuuint32_t *elementStrides,
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

    uint64_t size_weight_down_[rank] = {HIDDEN_DIM, FFN_DIM};
    uint64_t stride_weight_down_[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_down_[rank] = {TMA_LOAD_ONCE, FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX};
    uint32_t elem_stride_weight_down_[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_weight_down_ = cuTensorMapEncodeTiled(
        &tensor_map_weight_down_,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_ffn_down,                 // void *globalAddress,
        size_weight_down_,                       // const cuuint64_t *globalDim,
        stride_weight_down_,                     // const cuuint64_t *globalStrides,
        box_size_weight_down_,                   // const cuuint32_t *boxDim,
        elem_stride_weight_down_,                // const cuuint32_t *elementStrides,
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

    int wmup = 1;
    int test = 0;
    for (int i = 0; i < wmup; i++) {
        single_decode<<<grid, block, max_shmem_size>>>(
            d_output,
            d_input,
            global_reduce,
            d_rms_input,
            d_rms_attn,
            d_cos,
            d_sin,
            tensor_map_weight,
            tensor_map_k_cache,
            tensor_map_v_cache,
            tensor_map_weight_o,
            tensor_map_weight_gate_up,
            tensor_map_weight_gate_up_,
            tensor_map_weight_down,
            tensor_map_weight_down_
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
        single_decode<<<grid, block, max_shmem_size>>>(
            d_output,
            d_input,
            global_reduce,
            d_rms_input,
            d_rms_attn,
            d_cos,
            d_sin,
            tensor_map_weight,
            tensor_map_k_cache,
            tensor_map_v_cache,
            tensor_map_weight_o,
            tensor_map_weight_gate_up,
            tensor_map_weight_gate_up_,
            tensor_map_weight_down,
            tensor_map_weight_down_
        );
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / test * 1e3 << " us" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < HIDDEN_DIM; i++)
    //     printf("%f, ", __half2float(h_output[i]));
    return 0;
}