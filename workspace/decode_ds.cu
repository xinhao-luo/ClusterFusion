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
#include "dsm.cuh"
// nvcc --generate-code=arch=compute_90a,code=sm_90a -O3 -std=c++17 -lcuda decode_ds.cu -o test && ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./test

#define NOPE_HEAD_DIM 128    
#define ROPE_HEAD_DIM 64    
#define HEAD_DIM (ROPE_HEAD_DIM + NOPE_HEAD_DIM) 
#define MLA_HEAD_DIM (KV_LORA_RANK + ROPE_HEAD_DIM)
#define KV_LORA_RANK 512    
#define HEAD_NUM 16     
#define HIDDEN_DIM 2048 
#define SEQ_LEN 4096

#define NUM_WARPS 4 // 4 8 16 32
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE) 
#define CLUSTER_SIZE 4 // 2 4 8 16
#define NUM_PER_THREAD 8
#define NUM_ROW_PER_WARP (NOPE_HEAD_DIM / NUM_WARPS) 
#define NUM_THREAD_PER_ROW (WARP_SIZE / NUM_ROW_PER_WARP) 
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW) 
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)
#define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) 

#define TMA_LOAD_ONCE 64 // 8 16 32 64 128 256
#define TMA_LOAD_ONCE_NUM (TMA_LOAD_ONCE * NOPE_HEAD_DIM)
#define TMA_LOAD_ONCE_SIZE (TMA_LOAD_ONCE_NUM * sizeof(half))
#define TMA_LOAD_ONCE_ATTN (TMA_LOAD_ONCE / 4) // 16
#define TMA_LOAD_ONCE_NUM_ATTN (TMA_LOAD_ONCE_ATTN * KV_LORA_RANK) // 16 * 512
#define TMA_LOAD_ONCE_SIZE_ATTN (TMA_LOAD_ONCE_NUM_ATTN * sizeof(half))

#define NUM_ROW_PER_WARP_2 (TMA_LOAD_ONCE / NUM_WARPS) 
#define NUM_THREAD_PER_ROW_2 (WARP_SIZE / NUM_ROW_PER_WARP_2) 
#define NUM_PER_ROW_2 (NUM_PER_THREAD * NUM_THREAD_PER_ROW_2) 
#define NUM_THREAD_PER_ROW_3 (KV_LORA_RANK / NUM_PER_THREAD) // 64
#define NUM_ROW_PER_BLOCK (BLOCK_SIZE / NUM_THREAD_PER_ROW_3) // 2
#define DEC_TILE (TMA_LOAD_ONCE_ATTN / NUM_ROW_PER_BLOCK) // 8

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
    half* w_rms_input,// hidden_dim
    half* w_rms_ckv,// kv_lora_rank
    float* cos,       // head_dim
    float* sin,       // head_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_q, // hidden_dim * (head_num * nope_head_dim)
    const __grid_constant__ CUtensorMap tensor_map_weight_q_pe, // hidden_dim * (head_num * rope_head_dim)
    const __grid_constant__ CUtensorMap tensor_map_weight_uk, // nope_head_dim * (head_num * kv_lora_rank)
    const __grid_constant__ CUtensorMap tensor_map_weight_kv, // hidden_dim * kv_lora_rank
    const __grid_constant__ CUtensorMap tensor_map_weight_k_pe, // hidden_dim * rope_head_dim
    const __grid_constant__ CUtensorMap tensor_map_kv_cache, // seqlen * mla_head_dim
    const __grid_constant__ CUtensorMap tensor_map_kv_cache_, // seqlen * mla_head_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_uv, // kv_lora_rank * (head_num * nope_head_dim)
    const __grid_constant__ CUtensorMap tensor_map_weight_o // (head_num * nope_head_dim) * hidden_dim
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; 
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t tile_row = tid / NUM_THREAD_PER_ROW_3;
    const uint32_t tile_col = tid % NUM_THREAD_PER_ROW_3;

    // Init shared memory
    __shared__ __align__(16) half input_shmem[DIM_PER_BLOCK];
    __shared__ float reduction[NUM_WARPS];
    __shared__ float cluster_local_sum, cluster_local_max;
    __shared__ alignas(128) half weight[2 * TMA_LOAD_ONCE * NOPE_HEAD_DIM];
    __shared__ __align__(16) half local_qkv[MLA_HEAD_DIM * 2];

    // Init registers
    float local_sum = 0.0, eps = 1e-6, rms_rcp = 0.0, tmp = 0.0, local_max = 0.0, pre_max = 0.0, scale = 0.0, softmax_scale = __frsqrt_rn(HEAD_DIM);
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD], reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    half2 q_rope, q_rope_1, k_rope, k_rope_1;
    float2 cos_reg, sin_reg;
    uint32_t size;
    uint32_t src_addr, dst_addr, neighbor_dst_bar = 0;
    float __align__(16) qk[DEC_TILE];

    // Init barrier
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[2];
    barrier::arrival_token token[2];
    __shared__ uint64_t barrier;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    if (tid == 0) {
        init(&bar[0], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[1], blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    block.sync();

    // Precompute some indices
    uint input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    uint input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    uint weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2;
    uint input_idx_3 = tile_col * NUM_PER_THREAD;
    uint weight_idx_3 = tile_row * DEC_TILE;
    uint cluster_head_idx = head_id * NOPE_HEAD_DIM;
    uint cluster_head_pe_idx = head_id * ROPE_HEAD_DIM;
    uint cluster_head_uk_idx = head_id * KV_LORA_RANK + cluster_block_id * 128;
    uint cluster_block_st_idx = cluster_block_id * DIM_PER_BLOCK;
    uint cluster_block_st_uv_idx = cluster_block_id * (KV_LORA_RANK / CLUSTER_SIZE);

    // Load input to shared memory
    #pragma unroll
    for (int i = tid * 8; i < DIM_PER_BLOCK; i+=BLOCK_SIZE * 8) {
        *(uint4*)(&input_shmem[i]) = *(uint4*)(&input[cluster_block_st_idx + i]);
    }
    block.sync();

    // RMSNorm
    // for (int d = tid * 4; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 4) { 
    //     *(uint64_t*)(&reg_input[0]) = *(uint64_t*)(&input_shmem[d]);
    //     for (int di = 0; di < 4; di++)
    //         local_sum += __half2float(reg_input[di]) * __half2float(reg_input[di]);
    // }
    // #pragma unroll
    // for (int mask = 16; mask > 0; mask >>= 1) {
    //     local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    // }
    // if (lane_id == 0){
    //     reduction[warp_id] = local_sum;
    // }
    // block.sync(); 
    // if (tid < NUM_WARPS) 
    //     local_sum = reduction[tid];
    // #pragma unroll
    // for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
    //     local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    // } 
    // block.sync();
    // if (tid == 0)
    //     cluster_local_sum = local_sum;
    // cluster.sync();
    // // DSM Ring All-reduce
    // for (int i = 1; i < cluster.num_blocks() - 1; i++) {
    //     if (tid == 0) {
    //         local_sum = cluster_local_sum;
    //         int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
    //         dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);  
    //     }
    //     cluster.sync();
    //     if (tid == 0) {
    //         atomicAdd(dst_shmem, local_sum);
    //     }
    //     cluster.sync();
    // }
    // rms_rcp = __frsqrt_rn(cluster_local_sum / HIDDEN_DIM + eps);
    // for (int d = tid * 4; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 4) { 
    //     *(uint64_t*)(&reg_weight[0]) = *(uint64_t*)(&w_rms_input[cluster_block_st_idx + d]);
    //     for (int i = 0; i < 4; i++) {
    //         reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
    //     }
    //     *(uint64_t*)(&input_shmem[d]) = *(uint64_t*)(&reg_input[0]);
    // }
    // block.sync();

    // Compute input @ w_q
    // Preload w_q
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_q, cluster_head_idx, cluster_block_st_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_q, cluster_head_idx, cluster_block_st_idx + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * NOPE_HEAD_DIM + weight_idx]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (DIM_PER_BLOCK - TMA_LOAD_ONCE) + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * NOPE_HEAD_DIM + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // Compute input @ w_q_rope
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_q_pe, cluster_head_pe_idx, cluster_block_st_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    tmp = 0.0;
    for (int id = 1; id < DIM_PER_BLOCK / 128; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_q_pe, cluster_head_pe_idx, cluster_block_st_idx + id * 128, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < 128; i+=NUM_PER_ROW_2) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx_2 + (id - 1) * 128 + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx_2 + i + d) * ROPE_HEAD_DIM + weight_idx_2]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < 128; i+=NUM_PER_ROW_2) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx_2 + (DIM_PER_BLOCK - 128) + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx_2 + i + d) * ROPE_HEAD_DIM + weight_idx_2]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        local_qkv[KV_LORA_RANK + warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2] = __float2half(tmp);
    }
    block.sync();

    // Compute input @ w_kv
    for (int s = 0; s < KV_LORA_RANK / 128; s++) {
        tmp = 0.0;
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_kv, s * 128, cluster_block_st_idx, bar[0]);
            token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[0] = bar[0].arrive();
        }

        for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_kv, s * 128, cluster_block_st_idx + id * TMA_LOAD_ONCE, bar[id % 2]);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
            } else {
                token[id % 2] = bar[id % 2].arrive();
            }
            bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
                }
            }
        }
        bar[1].wait(std::move(token[1]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (DIM_PER_BLOCK - TMA_LOAD_ONCE) + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
            }
        }
        if (lane_id % NUM_THREAD_PER_ROW == 0) {
            local_qkv[MLA_HEAD_DIM + s * 128 + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
        }
    }
    block.sync();

    // Compute input @ w_k_pe
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_k_pe, 0, cluster_block_st_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    tmp = 0.0;
    for (int id = 1; id < DIM_PER_BLOCK / 128; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_k_pe, 0, cluster_block_st_idx + id * 128, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < 128; i+=NUM_PER_ROW_2) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx_2 + (id - 1) * 128 + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx_2 + i + d) * ROPE_HEAD_DIM + weight_idx_2]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < 128; i+=NUM_PER_ROW_2) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx_2 + (DIM_PER_BLOCK - 128) + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx_2 + i + d) * ROPE_HEAD_DIM + weight_idx_2]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2] = __float2half(tmp);
    }
    block.sync();

    // DSM All-reduce
    size = MLA_HEAD_DIM * 2 * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    dsm_ring_allreduce<CLUSTER_SIZE, Stage::LINEAR_DEEPSEEK>(
        size, tid, BLOCK_SIZE * 8, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, weight);

    // Compute partial RoPE
    // if (tid < ROPE_HEAD_DIM / 2) {
    //     q_rope = *(half2*)(&local_qkv[KV_LORA_RANK + tid * 2]);
    //     k_rope = *(half2*)(&local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + tid * 2]);
    //     if (tid * 2 < ROPE_HEAD_DIM / 2) {
    //         q_rope_1 = *(half2*)(&local_qkv[KV_LORA_RANK + ROPE_HEAD_DIM / 2 + tid * 2]);
    //         k_rope_1 = *(half2*)(&local_qkv[MLA_HEAD_DIM + ROPE_HEAD_DIM / 2 + tid * 2]);
    //         cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
    //         sin_reg = {-sin[ROPE_HEAD_DIM / 2 + tid * 2], -sin[ROPE_HEAD_DIM / 2 + tid * 2 + 1]};
    //     } else {
    //         q_rope_1 = *(half2*)(&local_qkv[KV_LORA_RANK + tid * 2 - ROPE_HEAD_DIM / 2]);
    //         k_rope_1 = *(half2*)(&local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + tid * 2 - ROPE_HEAD_DIM / 2]);
    //         cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
    //         sin_reg = {sin[tid * 2 - ROPE_HEAD_DIM / 2], sin[tid * 2 + 1 - ROPE_HEAD_DIM / 2]};
    //     }
    //     *(half2*)(&local_qkv[KV_LORA_RANK + tid * 2]) = __hadd2(__hmul2(q_rope, __float22half2_rn(cos_reg)), __hmul2(q_rope_1, __float22half2_rn(sin_reg)));
    //     *(half2*)(&local_qkv[MLA_HEAD_DIM + KV_LORA_RANK + tid * 2]) = __hadd2(__hmul2(k_rope, __float22half2_rn(cos_reg)), __hmul2(k_rope_1, __float22half2_rn(sin_reg)));
    // }

    // RMSNorm
    local_sum = 0.0;
    for (int d = tid * 4; d < KV_LORA_RANK; d+=BLOCK_SIZE * 4) { 
        *(uint64_t*)(&reg_input[0]) = *(uint64_t*)(&local_qkv[MLA_HEAD_DIM + d]);
        for (int di = 0; di < 4; di++)
            local_sum += __half2float(reg_input[di]) * __half2float(reg_input[di]);
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
    block.sync();
    rms_rcp = __frsqrt_rn(cluster_local_sum / KV_LORA_RANK + eps);
    for (int d = tid * 4; d < KV_LORA_RANK; d+=BLOCK_SIZE * 4) { 
        *(uint64_t*)(&reg_weight[0]) = *(uint64_t*)(&w_rms_ckv[d]);
        for (int i = 0; i < 4; i++) {
            reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
        }
        *(uint64_t*)(&local_qkv[MLA_HEAD_DIM + d]) = *(uint64_t*)(&reg_input[0]);
    }
    block.sync();
    
    // Compute q @ w_uk
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_uk, cluster_head_uk_idx, 0, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    tmp = 0.0;
    for (int id = 1; id < NOPE_HEAD_DIM / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_uk, cluster_head_uk_idx, id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx + (NOPE_HEAD_DIM - TMA_LOAD_ONCE) + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[cluster_block_id * 128 + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // DSM All-gather
    size = 128 * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[cluster_block_id * 128]));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[cluster_block_id * 128]));
    dsm_ring_allreduce<CLUSTER_SIZE, Stage::QUK_DEEPSEEK>(
        size, tid, 0, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, local_qkv);

    // Compute flash-decoding
    local_sum = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = __float2half(0.0f);
    *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3]);
    block.sync();

    // Preload kv_cache
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_kv_cache, 0, cluster_block_id * KV_DIM_PER_BLOCK, bar[0]);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_ATTN / 2], &tensor_map_kv_cache_, KV_LORA_RANK / 2, cluster_block_id * KV_DIM_PER_BLOCK, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_ATTN);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_kv_cache, 0, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_ATTN + TMA_LOAD_ONCE_NUM_ATTN / 2], &tensor_map_kv_cache_, KV_LORA_RANK / 2, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        pre_max = local_max;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM_ATTN + (warp_id % 2) * TMA_LOAD_ONCE_NUM_ATTN / 2 + (weight_idx_3 + j) * KV_LORA_RANK / 2 + lane_id * NUM_PER_THREAD]);
            qk[j] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
            }
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
            }
            block.sync();
            if (lane_id == 0)
                reduction[warp_id] = qk[j];
            block.sync();
            if (tile_row == 1)
                qk[j] += reduction[2 + (warp_id + 1) % 2];
            else
                qk[j] += reduction[(warp_id + 1) % 2];
            qk[j] = qk[j] * softmax_scale;
            local_max = fmaxf(local_max, qk[j]);
        }
        scale = __expf(pre_max - local_max);
        local_sum *= scale;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            qk[j] = __expf(qk[j] - local_max);
            local_sum += qk[j];
        }
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++) {
            reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        }
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM_ATTN + (warp_id % 2) * TMA_LOAD_ONCE_NUM_ATTN / 2 + (weight_idx_3 + j) * KV_LORA_RANK / 2 + lane_id * NUM_PER_THREAD]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    pre_max = local_max;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        if (cluster_block_id == CLUSTER_SIZE - 1 && tile_row == 1 && j == DEC_TILE - 1)
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[MLA_HEAD_DIM + input_idx_3]);
        else
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[TMA_LOAD_ONCE_NUM_ATTN + (warp_id % 2) * TMA_LOAD_ONCE_NUM_ATTN / 2 + (weight_idx_3 + j) * KV_LORA_RANK / 2 + lane_id * NUM_PER_THREAD]);
        qk[j] = 0.0f;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
        }
        if (lane_id == 0)
            reduction[warp_id] = qk[j];
        block.sync();
        if (tile_row == 1)
            qk[j] += reduction[2 + (warp_id + 1) % 2];
        else
            qk[j] += reduction[(warp_id + 1) % 2];
        qk[j] = qk[j] * softmax_scale;
        local_max = fmaxf(local_max, qk[j]);
    }
    scale = __expf(pre_max - local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        qk[j] = __expf(qk[j] - local_max);
        local_sum += qk[j];
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
    }
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        if (cluster_block_id == CLUSTER_SIZE - 1 && tile_row == 1 && j == DEC_TILE - 1)
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[MLA_HEAD_DIM + input_idx_3]);
        else
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[TMA_LOAD_ONCE_NUM_ATTN + (warp_id % 2) * TMA_LOAD_ONCE_NUM_ATTN / 2 + (weight_idx_3 + j) * KV_LORA_RANK / 2 + lane_id * NUM_PER_THREAD]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
        }
    }
    block.sync();

    *(uint4*)(&weight[tile_row * MLA_HEAD_DIM + tile_col * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
    if (tid % NUM_THREAD_PER_ROW_3 == 0) {
        reduction[tile_row * 2] = local_max;
        reduction[tile_row * 2 + 1] = local_sum;
    }
    block.sync();
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = __float2half(0.0f);
    local_sum = 0.0, local_max = 0.0;
    #pragma unroll
    for(int j = 0; j < NUM_ROW_PER_BLOCK; j++) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * MLA_HEAD_DIM + tile_col * NUM_PER_THREAD]);
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = fmaxf(m, local_max);
        scale = __expf(m - local_max);
        s *= scale;
        local_sum = local_sum * __expf(pre_max - local_max) + s;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] = __hadd(__hmul(reg_reduce[d], __float2half(__expf(pre_max - local_max))), __hmul(reg_input[d], __float2half(scale)));
        }
    }
    block.sync();
    pre_max = local_max;
    if(tid == 0) {
        cluster_local_max = local_max;
    }
    cluster.sync();
    // DSM Ring All-reduce
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_max = cluster_local_max;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            dst_shmem = cluster.map_shared_rank(&cluster_local_max, dst_cta);  
        }
        cluster.sync();
        if (tid == 0) {
            *dst_shmem = fmaxf(*dst_shmem, local_max);
        }
        cluster.sync();
    }
    scale = __expf(pre_max - cluster_local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
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
            dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);  
        }
        cluster.sync();
        if (tid == 0) {
            atomicAdd(dst_shmem, local_sum);
        }
        cluster.sync();
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = __hmul(reg_reduce[j], __float2half(__frcp_rn(cluster_local_sum)));
    }
    if(tid < NUM_THREAD_PER_ROW_3) {
        *(uint4*)(&local_qkv[tid * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
    }
    block.sync();
    // DSM Ring All-reduce
    size = KV_LORA_RANK * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    dsm_ring_allreduce<CLUSTER_SIZE, Stage::ATTN_DEEPSEEK>(
        size, tid, KV_LORA_RANK, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, weight);
    
    // Compute output @ w_uv
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_uv, cluster_head_idx, cluster_block_st_uv_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    tmp = 0.0;
    for (int id = 1; id < 128 / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_uv, cluster_head_idx, cluster_block_st_uv_idx + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[cluster_block_st_uv_idx + input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[cluster_block_st_uv_idx + input_idx + (NOPE_HEAD_DIM - TMA_LOAD_ONCE) + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * 128 + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    size = NOPE_HEAD_DIM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    dsm_ring_allreduce<CLUSTER_SIZE, Stage::ATTN>(
        size, tid, NOPE_HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, weight);

    // Compute output @ w_o
    // Preload w_o
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_o, cluster_block_st_idx, cluster_head_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_o, cluster_block_st_idx + id * TMA_LOAD_ONCE, cluster_head_idx, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        tmp = 0.0;
        for (int j = 0; j < NOPE_HEAD_DIM; j+=NUM_PER_ROW_2) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2 + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM + (input_idx_2 + j + d) * TMA_LOAD_ONCE + weight_idx_2]));
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
        if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
            atomicAdd(&output[cluster_block_st_idx + weight_idx_2 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
        }
    }
    bar[1].wait(std::move(token[1]));
    tmp = 0.0;
    for (int j = 0; j < NOPE_HEAD_DIM; j+=NUM_PER_ROW_2) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2 + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx_2 + j + d) * TMA_LOAD_ONCE + weight_idx_2]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        atomicAdd(&output[cluster_block_st_idx + weight_idx_2 + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    }
}

int main(int argc, char** argv) {
    cudaFuncSetAttribute(single_decode, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    half *h_input, *d_input;
    half *h_ckv_cache, *d_ckv_cache;
    half *h_w_q_nope, *d_w_q_nope;
    half *h_w_q_pe, *d_w_q_pe;
    half *h_w_uk, *d_w_uk;
    half *h_w_kv, *d_w_kv;
    half *h_w_k_pe, *d_w_k_pe;
    half *h_w_uv, *d_w_uv;
    half *h_w_o, *d_w_o;
    half *h_rms_input, *d_rms_input;
    half *h_rms_ckv, *d_rms_ckv;
    float *h_cos, *d_cos;
    float *h_sin, *d_sin;
    h_input = new half[1 * HIDDEN_DIM];
    h_w_q_nope = new half[HIDDEN_DIM * (HEAD_NUM * NOPE_HEAD_DIM)];
    h_w_q_pe = new half[HIDDEN_DIM * (HEAD_NUM * ROPE_HEAD_DIM)];
    h_w_uk = new half[HEAD_NUM * NOPE_HEAD_DIM * KV_LORA_RANK];
    h_w_kv = new half[HIDDEN_DIM * KV_LORA_RANK];
    h_w_k_pe = new half[HIDDEN_DIM * ROPE_HEAD_DIM];
    h_w_uv = new half[HEAD_NUM * KV_LORA_RANK * NOPE_HEAD_DIM];
    h_w_o = new half[HIDDEN_DIM * HIDDEN_DIM];
    h_ckv_cache = new half[SEQ_LEN * MLA_HEAD_DIM];
    h_rms_input = new half[HIDDEN_DIM];
    h_rms_ckv = new half[KV_LORA_RANK];
    h_cos = new float[ROPE_HEAD_DIM];
    h_sin = new float[ROPE_HEAD_DIM];

    fill_matrix(h_input, 1 * HIDDEN_DIM);
    fill_matrix(h_w_q_nope, HIDDEN_DIM * (HEAD_NUM * NOPE_HEAD_DIM));
    fill_matrix(h_w_q_pe, HIDDEN_DIM * (HEAD_NUM * ROPE_HEAD_DIM));
    fill_matrix(h_w_uk, HEAD_NUM * NOPE_HEAD_DIM * KV_LORA_RANK);
    fill_matrix(h_w_kv, HIDDEN_DIM * KV_LORA_RANK);
    fill_matrix(h_w_k_pe, HIDDEN_DIM * ROPE_HEAD_DIM);
    fill_matrix(h_w_uv, HEAD_NUM * KV_LORA_RANK * NOPE_HEAD_DIM);
    fill_matrix(h_w_o, HIDDEN_DIM * HIDDEN_DIM);
    fill_matrix(h_ckv_cache, SEQ_LEN * MLA_HEAD_DIM);
    fill_matrix(h_rms_input, HIDDEN_DIM);
    fill_matrix(h_rms_ckv, KV_LORA_RANK);

    // Init cos, sin used in RoPE
    int encode_point_offset = 0;
    float rope_scale = 1;
    float rope_theta = 500000;
    for (int j = 0; j < ROPE_HEAD_DIM; j++) {
        float inv_freq =(encode_point_offset / rope_scale) / (std::pow(rope_theta, float(2 * (j % (ROPE_HEAD_DIM / 2))) / float(ROPE_HEAD_DIM)));
        h_cos[j] = std::cos(inv_freq);
        h_sin[j] = std::sin(inv_freq);
    }

    cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * 1 * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_q_nope), sizeof(half) * HIDDEN_DIM * (HEAD_NUM * NOPE_HEAD_DIM));
    cudaMalloc(reinterpret_cast<void**>(&d_w_q_pe), sizeof(half) * HIDDEN_DIM * (HEAD_NUM * ROPE_HEAD_DIM));
    cudaMalloc(reinterpret_cast<void**>(&d_w_uk), sizeof(half) * HEAD_NUM * NOPE_HEAD_DIM * KV_LORA_RANK);
    cudaMalloc(reinterpret_cast<void**>(&d_w_kv), sizeof(half) * HIDDEN_DIM * KV_LORA_RANK);
    cudaMalloc(reinterpret_cast<void**>(&d_w_k_pe), sizeof(half) * HIDDEN_DIM * ROPE_HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_uv), sizeof(half) * HEAD_NUM * KV_LORA_RANK * NOPE_HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_o), sizeof(half) * HIDDEN_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_input), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_ckv), sizeof(half) * KV_LORA_RANK);
    cudaMalloc(reinterpret_cast<void**>(&d_ckv_cache), sizeof(half) * SEQ_LEN * MLA_HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_cos), sizeof(float) * ROPE_HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_sin), sizeof(float) * ROPE_HEAD_DIM);
    
    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_q_nope), h_w_q_nope, sizeof(half) * HIDDEN_DIM * (HEAD_NUM * NOPE_HEAD_DIM), cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_q_pe), h_w_q_pe, sizeof(half) * HIDDEN_DIM * (HEAD_NUM * ROPE_HEAD_DIM), cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_uk), h_w_uk, sizeof(half) * HEAD_NUM * NOPE_HEAD_DIM * KV_LORA_RANK, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_kv), h_w_kv, sizeof(half) * HIDDEN_DIM * KV_LORA_RANK, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_k_pe), h_w_k_pe, sizeof(half) * HIDDEN_DIM * ROPE_HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_uv), h_w_uv, sizeof(half) * HEAD_NUM * KV_LORA_RANK * NOPE_HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_o), h_w_o, sizeof(half) * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_input), h_rms_input, sizeof(half) * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_ckv), h_rms_ckv, sizeof(half) * KV_LORA_RANK, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ckv_cache), h_ckv_cache, sizeof(half) * SEQ_LEN * MLA_HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_cos), h_cos, sizeof(float) * ROPE_HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_sin), h_sin, sizeof(float) * ROPE_HEAD_DIM, cudaMemcpyHostToDevice);

    half* h_output, *d_output;
    h_output = new half[1 * HIDDEN_DIM];
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * 1 * HIDDEN_DIM);

    CUtensorMap tensor_map_weight_q{};
    CUtensorMap tensor_map_weight_q_pe{};
    CUtensorMap tensor_map_weight_uk{};
    CUtensorMap tensor_map_weight_kv{};
    CUtensorMap tensor_map_weight_k_pe{};
    CUtensorMap tensor_map_kv_cache{};
    CUtensorMap tensor_map_kv_cache_{};
    CUtensorMap tensor_map_weight_uv{};
    CUtensorMap tensor_map_weight_o{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {HEAD_NUM * NOPE_HEAD_DIM, HIDDEN_DIM};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {HEAD_NUM * NOPE_HEAD_DIM * sizeof(half)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {NOPE_HEAD_DIM, TMA_LOAD_ONCE};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride[rank] = {1, 1};
    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map_weight_q,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_w_q_nope,                 // void *globalAddress,
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

    uint64_t size_w_q_rope[rank] = {HEAD_NUM * ROPE_HEAD_DIM, HIDDEN_DIM};
    uint64_t stride_w_q_rope[rank - 1] = {HEAD_NUM * ROPE_HEAD_DIM * sizeof(half)};
    uint32_t box_size_w_q_rope[rank] = {TMA_LOAD_ONCE, 128};
    uint32_t elem_stride_w_q_rope[rank] = {1, 1};
    CUresult res_2 = cuTensorMapEncodeTiled(
        &tensor_map_weight_q_pe,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_w_q_pe,                 // void *globalAddress,
        size_w_q_rope,                       // const cuuint64_t *globalDim,
        stride_w_q_rope,                     // const cuuint64_t *globalStrides,
        box_size_w_q_rope,                   // const cuuint32_t *boxDim,
        elem_stride_w_q_rope,                // const cuuint32_t *elementStrides,
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

    uint64_t size_w_uk[rank] = {HEAD_NUM * KV_LORA_RANK, NOPE_HEAD_DIM};
    uint64_t stride_w_uk[rank - 1] = {HEAD_NUM * KV_LORA_RANK * sizeof(half)};
    uint32_t box_size_w_uk[rank] = {128, TMA_LOAD_ONCE};
    uint32_t elem_stride_w_uk[rank] = {1, 1};
    CUresult res_3 = cuTensorMapEncodeTiled(
        &tensor_map_weight_uk,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_w_uk,                 // void *globalAddress,
        size_w_uk,                       // const cuuint64_t *globalDim,
        stride_w_uk,                     // const cuuint64_t *globalStrides,
        box_size_w_uk,                   // const cuuint32_t *boxDim,
        elem_stride_w_uk,                // const cuuint32_t *elementStrides,
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

    uint64_t size_w_kv[rank] = {KV_LORA_RANK, HIDDEN_DIM};
    uint64_t stride_w_kv[rank - 1] = {KV_LORA_RANK * sizeof(half)};
    uint32_t box_size_w_kv[rank] = {128, TMA_LOAD_ONCE};
    uint32_t elem_stride_w_kv[rank] = {1, 1};
    CUresult res_4 = cuTensorMapEncodeTiled(
        &tensor_map_weight_kv,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_w_kv,                 // void *globalAddress,
        size_w_kv,                       // const cuuint64_t *globalDim,
        stride_w_kv,                     // const cuuint64_t *globalStrides,
        box_size_w_kv,                   // const cuuint32_t *boxDim,
        elem_stride_w_kv,                // const cuuint32_t *elementStrides,
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

    uint64_t size_w_k_pe[rank] = {ROPE_HEAD_DIM, HIDDEN_DIM};
    uint64_t stride_w_k_pe[rank - 1] = {ROPE_HEAD_DIM * sizeof(half)};
    uint32_t box_size_w_k_pe[rank] = {TMA_LOAD_ONCE, 128};
    uint32_t elem_stride_w_k_pe[rank] = {1, 1};
    CUresult res_5 = cuTensorMapEncodeTiled(
        &tensor_map_weight_k_pe,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_w_k_pe,                 // void *globalAddress,
        size_w_k_pe,                       // const cuuint64_t *globalDim,
        stride_w_k_pe,                     // const cuuint64_t *globalStrides,
        box_size_w_k_pe,                   // const cuuint32_t *boxDim,
        elem_stride_w_k_pe,                // const cuuint32_t *elementStrides,
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

    uint64_t size_kv_cache[rank] = {MLA_HEAD_DIM, SEQ_LEN};
    uint64_t stride_kv_cache[rank - 1] = {MLA_HEAD_DIM * sizeof(half)};
    uint32_t box_size_kv_cache[rank] = {256, TMA_LOAD_ONCE_ATTN};
    uint32_t elem_stride_kv_cache[rank] = {1, 1};
    CUresult res_6 = cuTensorMapEncodeTiled(
        &tensor_map_kv_cache,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_ckv_cache,                 // void *globalAddress,
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

    CUresult res_7 = cuTensorMapEncodeTiled(
        &tensor_map_kv_cache_,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_ckv_cache,                 // void *globalAddress,
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

    uint64_t size_w_uv[rank] = {HEAD_NUM * NOPE_HEAD_DIM, KV_LORA_RANK};
    uint64_t stride_w_uv[rank - 1] = {HEAD_NUM * NOPE_HEAD_DIM * sizeof(half)};
    uint32_t box_size_w_uv[rank] = {128, TMA_LOAD_ONCE};
    uint32_t elem_stride_w_uv[rank] = {1, 1};
    CUresult res_8 = cuTensorMapEncodeTiled(
        &tensor_map_weight_uv,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_w_uv,                 // void *globalAddress,
        size_w_uv,                       // const cuuint64_t *globalDim,
        stride_w_uv,                     // const cuuint64_t *globalStrides,
        box_size_w_uv,                   // const cuuint32_t *boxDim,
        elem_stride_w_uv,                // const cuuint32_t *elementStrides,
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

    uint64_t size_w_o[rank] = {HIDDEN_DIM, HIDDEN_DIM};
    uint64_t stride_w_o[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_w_o[rank] = {TMA_LOAD_ONCE, NOPE_HEAD_DIM};
    uint32_t elem_stride_w_o[rank] = {1, 1};
    CUresult res_9 = cuTensorMapEncodeTiled(
        &tensor_map_weight_o,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        d_w_o,                 // void *globalAddress,
        size_w_o,                       // const cuuint64_t *globalDim,
        stride_w_o,                     // const cuuint64_t *globalStrides,
        box_size_w_o,                   // const cuuint32_t *boxDim,
        elem_stride_w_o,                // const cuuint32_t *elementStrides,
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

    cudaDeviceSynchronize();

    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    int wmup = 0;
    int test = 1;
    for (int i = 0; i < wmup; i++) {
        single_decode<<<grid, block>>>(
            d_output,
            d_input,
            d_rms_input,
            d_rms_ckv,
            d_cos,
            d_sin,
            tensor_map_weight_q,
            tensor_map_weight_q_pe,
            tensor_map_weight_uk,
            tensor_map_weight_kv,
            tensor_map_weight_k_pe,
            tensor_map_kv_cache,
            tensor_map_kv_cache_,
            tensor_map_weight_uv,
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
            d_rms_input,
            d_rms_ckv,
            d_cos,
            d_sin,
            tensor_map_weight_q,
            tensor_map_weight_q_pe,
            tensor_map_weight_uk,
            tensor_map_weight_kv,
            tensor_map_weight_k_pe,
            tensor_map_kv_cache,
            tensor_map_kv_cache_,
            tensor_map_weight_uv,
            tensor_map_weight_o
        );
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / test * 1e3 << " us" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < HIDDEN_DIM; i++)
    //     printf("%f, ", __half2float(h_output[0]));
    return 0;
}