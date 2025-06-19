#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <iostream>
#include "dsm.cuh"
#include "config_linear-attn.h"
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

// CUDA_VISIBLE_DEVICES=1 nvcc --generate-code=arch=compute_90a,code=sm_90a -O3 -std=c++17 -lcuda linear-attn.cu -o test && ./test

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) LlamaDecoderLayerKernel(
    half* output, // 1 * hidden_dim
    half* input,  // 1 * hidden_dim
    const __grid_constant__ CUtensorMap tensor_map, // 3 * hidden_dim * hidden_dim
    const __grid_constant__ CUtensorMap tensor_map_k_cache, // seqlen * head_num * head_dim
    const __grid_constant__ CUtensorMap tensor_map_v_cache, // seqlen * head_num * head_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_o // hidden_dim * hidden_dim
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; 
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t tile_row = tid / NUM_THREAD_PER_ROW_2;
    const uint32_t tile_col = tid % NUM_THREAD_PER_ROW_2;

    // Init shared memory
    __shared__ __align__(16) half input_shmem[DIM_PER_BLOCK];
    __shared__ alignas(128) half weight[2 * TMA_LOAD_ONCE * HEAD_DIM];
    __shared__ __align__(16) half local_qkv[3 * HEAD_DIM];
    __shared__ float reduction[2 * NUM_PER_ROW_2];
    __shared__ float cluster_local_sum, cluster_local_max;

    // Init registers
    float local_sum = 0.0, tmp = 0.0, local_max = 0.0, pre_max = 0.0, scale = 0.0, softmax_scale = __frsqrt_rn(HEAD_DIM);
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD], reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    uint32_t size;
    uint32_t src_addr, dst_addr, neighbor_dst_bar = 0;
    float __align__(16) qk[DEC_TILE];

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
    uint input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    uint input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    uint weight_idx_2 = warp_id * NUM_PER_ROW_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
    uint input_idx_3 = (lane_id % NUM_THREAD_PER_ROW_3) * NUM_PER_THREAD;
    uint weight_idx_3 = warp_id * NUM_ROW_PER_WARP_3 + lane_id / NUM_THREAD_PER_ROW_3;
    uint cluster_block_st_id = cluster_block_id * DIM_PER_BLOCK;
    uint cluster_head_idx = head_id * HEAD_DIM;

    // Load input to shared memory
    #pragma unroll
    for (int i = tid * 8; i < DIM_PER_BLOCK; i+=BLOCK_SIZE * 8) {
        *(uint4*)(&input_shmem[i]) = *(uint4*)(&input[cluster_block_st_id + i]);
    }
    block.sync();

    // Compute input @ w_q
    // Preload weight_q
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_head_idx, cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        // bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_head_idx, cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // Compute input @ w_k
    // Preload weight_k
    tmp = 0.0;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_head_idx, HIDDEN_DIM + cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        // bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_head_idx, HIDDEN_DIM + cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[HEAD_DIM + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // Compute input @ w_v
    // Preload weight_v
    tmp = 0.0;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_head_idx, HIDDEN_DIM * 2 + cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        // bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_head_idx, HIDDEN_DIM * 2 + cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]));
            }
        }
    }
    bar[1].wait(std::move(token[1]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx + i + d) * HEAD_DIM + weight_idx]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_qkv[HEAD_DIM * 2 + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // DSM Ring All-reduce
    size = (HEAD_DIM * 3) * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    dsm_ring_allreduce<CLUSTER_SIZE, Stage::LINEAR>(
        size, tid, HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, weight);

    // Compute flash-decoding
    local_sum = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = __float2half(0.0f);
    *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2]);
    block.sync();

    // Preload kv_cache
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_k_cache, cluster_head_idx, cluster_block_id * KV_DIM_PER_BLOCK, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_v_cache, cluster_head_idx, cluster_block_id * KV_DIM_PER_BLOCK, bar[2]);
        token[2] = cuda::device::barrier_arrive_tx(bar[2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
    } else {
        token[0] = bar[0].arrive();
        token[2] = bar[2].arrive();
    }

    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_k_cache, cluster_head_idx, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        pre_max = local_max;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            qk[j] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
            }
            qk[j] = qk[j] * softmax_scale;
            local_max = max(local_max, qk[j]);
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
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_v_cache, cluster_head_idx, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[2 + id % 2]);
            token[2 + id % 2] = cuda::device::barrier_arrive_tx(bar[2 + id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[2 + id % 2] = bar[2 + id % 2].arrive();
        }
        bar[2 + (id - 1) % 2].wait(std::move(token[2 + (id - 1) % 2]));
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
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
        if (cluster_block_id == CLUSTER_SIZE - 1 && warp_id == NUM_WARPS - 1 && lane_id / NUM_THREAD_PER_ROW_2 == 1 && j == DEC_TILE - 1)
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[HEAD_DIM + input_idx_2]);
        else
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        qk[j] = 0.0f;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
        }
        qk[j] = qk[j] * softmax_scale;
        local_max = max(local_max, qk[j]);
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
    bar[3].wait(std::move(token[3]));
    for (int j = 0; j < DEC_TILE; j++) {
        if (cluster_block_id == CLUSTER_SIZE - 1 && warp_id == NUM_WARPS - 1 && lane_id / NUM_THREAD_PER_ROW_2 == 1 && j == DEC_TILE - 1) 
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_2]);
        else
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
        }
    }
    block.sync();

    *(uint4*)(&weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        reduction[tile_row * 2] = local_max;
        reduction[tile_row * 2 + 1] = local_sum;
    }
    block.sync();
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = __float2half(0.0f);
    local_sum = 0.0, local_max = 0.0;
    #pragma unroll
    for(int j = 0; j < NUM_PER_ROW_2; j++) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * HEAD_DIM + tile_col * NUM_PER_THREAD]);
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = max(m, local_max);
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
    if(tid < NUM_THREAD_PER_ROW_2) {
        *(uint4*)(&local_qkv[HEAD_DIM + HEAD_DIM + tid * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
    }
    block.sync();

    // DSM Ring-All reduce
    size = HEAD_DIM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[HEAD_DIM + HEAD_DIM]));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    dsm_ring_allreduce<CLUSTER_SIZE, Stage::ATTN>(
        size, tid, HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, &local_qkv[HEAD_DIM + HEAD_DIM], weight);

    // Compute output @ w_o
    // Preload w_o
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_o, cluster_block_st_id, cluster_head_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_o, cluster_block_st_id + id * TMA_LOAD_ONCE, cluster_head_idx, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        tmp = 0.0;
        for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_3) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[HEAD_DIM + HEAD_DIM + input_idx_3 + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(__hmul(reg_input[d], weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]));
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
    for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_3) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[HEAD_DIM + HEAD_DIM + input_idx_3 + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]));
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
    cudaFuncSetAttribute(LlamaDecoderLayerKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 16);
    // Init input
    half *h_input, *d_input;
    half *h_k_cache, *d_k_cache;
    half *h_v_cache, *d_v_cache;
    half *h_w_qkv, *d_w_qkv;
    half *h_w_o, *d_w_o;
    h_input = new half[BATCH_SIZE * HIDDEN_DIM];
    h_w_qkv = new half[3 * HIDDEN_DIM * HIDDEN_DIM];
    h_w_o = new half[HIDDEN_DIM * HIDDEN_DIM];
    h_k_cache = new half[BATCH_SIZE * SEQ_LEN * HEAD_NUM * HEAD_DIM];
    h_v_cache = new half[BATCH_SIZE * SEQ_LEN * HEAD_NUM * HEAD_DIM];

    fill_matrix(h_input, BATCH_SIZE * HIDDEN_DIM);
    fill_matrix(h_w_qkv, 3 * HIDDEN_DIM * HIDDEN_DIM);
    fill_matrix(h_w_o, HIDDEN_DIM * HIDDEN_DIM);
    fill_matrix(h_k_cache, BATCH_SIZE * SEQ_LEN * HEAD_NUM * HEAD_DIM);
    fill_matrix(h_v_cache, BATCH_SIZE * SEQ_LEN * HEAD_NUM * HEAD_DIM);

    cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * BATCH_SIZE * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_qkv), sizeof(half) * 3 * HIDDEN_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_o), sizeof(half) * HIDDEN_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_k_cache), sizeof(half) * BATCH_SIZE * SEQ_LEN * HEAD_NUM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_v_cache), sizeof(half) * BATCH_SIZE * SEQ_LEN * HEAD_NUM * HEAD_DIM);

    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * BATCH_SIZE *  HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_qkv), h_w_qkv, sizeof(half) * 3 * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_o), h_w_o, sizeof(half) * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_k_cache), h_k_cache, sizeof(half) * BATCH_SIZE * SEQ_LEN * HEAD_NUM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_v_cache), h_v_cache, sizeof(half) * BATCH_SIZE * SEQ_LEN * HEAD_NUM * HEAD_DIM, cudaMemcpyHostToDevice);

    half* h_output, *d_output;
    h_output = new half[BATCH_SIZE * HIDDEN_DIM];
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * BATCH_SIZE * HIDDEN_DIM);
    
    CUtensorMap tensor_map_weight{};
    CUtensorMap tensor_map_k_cache{};
    CUtensorMap tensor_map_v_cache{};
    CUtensorMap tensor_map_weight_o{};
    
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {HIDDEN_DIM, 3 * HIDDEN_DIM};
    uint64_t stride[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    uint32_t elem_stride[rank] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map_weight,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        d_w_qkv,                
        size,                      
        stride,                     
        box_size,                   
        elem_stride,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_k_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_k_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_k_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_k_cache[rank] = {1, 1};

    CUresult res_k_cache = cuTensorMapEncodeTiled(
        &tensor_map_k_cache,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        d_k_cache,                
        size_k_cache,                      
        stride_k_cache,                     
        box_size_k_cache,                   
        elem_stride_k_cache,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_v_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_v_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_v_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_v_cache[rank] = {1, 1};

    CUresult res_v_cache = cuTensorMapEncodeTiled(
        &tensor_map_v_cache,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        d_v_cache,                
        size_v_cache,                      
        stride_v_cache,                     
        box_size_v_cache,                   
        elem_stride_v_cache,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_weight_o[rank] = {HIDDEN_DIM, HIDDEN_DIM};
    uint64_t stride_weight_o[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_o[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_weight_o[rank] = {1, 1};

    CUresult res_weight_o = cuTensorMapEncodeTiled(
        &tensor_map_weight_o,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        d_w_o,                
        size_weight_o,                      
        stride_weight_o,                     
        box_size_weight_o,                   
        elem_stride_weight_o,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    int wmup = 100;
    int test = 100;
    for (int i = 0; i < wmup; i++) {
        LlamaDecoderLayerKernel<<<grid, block>>>(
            d_output,
            d_input,
            tensor_map_weight,
            tensor_map_k_cache,
            tensor_map_v_cache,
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
        LlamaDecoderLayerKernel<<<grid, block>>>(
            d_output,
            d_input,
            tensor_map_weight,
            tensor_map_k_cache,
            tensor_map_v_cache,
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
    //     printf("%f, ", __half2float(h_output[i]));
    // printf("\n");
    return 0;
}