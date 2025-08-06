#include <cuda/barrier>
#include <cudaTypedefs.h>
#include "../../dsm.cuh"
#include "config.h"
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) LlamaDecoderLayerKernel(
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
    // __shared__ __align__(16) half input_shmem[DIM_PER_BLOCK];
    // __shared__ float reduction[2 * DIM_BLOCK_REDUCE];
    // __shared__ alignas(128) half weight[2 * TMA_LOAD_ONCE * MAX_SMEM_DIM];
    // __shared__ __align__(16) half local_qkv[MAX_SMEM_DIM + MAX_SMEM_DIM + HEAD_DIM];
    extern __shared__ uint8_t shmem_base[];
    half* input_shmem = reinterpret_cast<half*>(shmem_base);
    float* reduction  = reinterpret_cast<float*>(shmem_base + DIM_PER_BLOCK * sizeof(half));
    half* weight      = reinterpret_cast<half*>((uintptr_t)(shmem_base + DIM_PER_BLOCK * sizeof(half) + 2 * DIM_BLOCK_REDUCE * sizeof(float)) + 127 & ~127);
    half* local_qkv   = reinterpret_cast<half*>((uintptr_t)(weight + 2 * TMA_LOAD_ONCE * MAX_SMEM_DIM) + 127 & ~127);

    __shared__ float cluster_local_sum, cluster_local_max;

    // Init registers
    float local_sum = 0.0, eps = 1e-6, rms_rcp = 0.0, tmp = 0.0, local_max = 0.0, pre_max = 0.0, scale = 0.0, softmax_scale = __frsqrt_rn(HEAD_DIM);
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD], reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    half2 q_rope, q_rope_1, k_rope, k_rope_1;
    float2 cos_reg, sin_reg;
    uint32_t size;
    uint32_t src_addr, dst_addr, neighbor_dst_bar = 0;
    float __align__(16) qk[DEC_TILE];
    float tmp_ffn[FFN_DIM_PER_CLUSTER / HEAD_DIM];

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
    uint weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
    uint input_idx_3 = (lane_id % NUM_THREAD_PER_ROW_3) * NUM_PER_THREAD;
    uint weight_idx_3 = warp_id * NUM_ROW_PER_WARP_3 + lane_id / NUM_THREAD_PER_ROW_3;
    uint cluster_block_st_id = cluster_block_id * DIM_PER_BLOCK;
    uint cluster_head_idx = head_id * HEAD_DIM;
    uint cluster_head_ffn_idx = head_id * FFN_DIM_PER_CLUSTER;

    // RMSNorm
    for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[cluster_block_st_id + d]);
        for (int di = 0; di < 8; di++)
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
    cluster.sync();
    // DSM Ring All-reduce
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
    rms_rcp = __frsqrt_rn(cluster_local_sum / HIDDEN_DIM + eps);
    for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&w_rms_input[cluster_block_st_id + d]);
        for (int i = 0; i < 8; i++) {
            reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
        }
        *(uint4*)(&input_shmem[d]) = *(uint4*)(&reg_input[0]);
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

    /*
      TODO: RoPE need debug  
    */ 

    // Compute RoPE
    // if (tid < HEAD_DIM / 2) {
    //     q_rope = *(half2*)(&local_qkv[tid * 2]);
    //     k_rope = *(half2*)(&local_qkv[HEAD_DIM + tid * 2]);
    //     if (tid * 2 < HEAD_DIM / 2) {
    //         q_rope_1 = *(half2*)(&local_qkv[HEAD_DIM / 2 + tid * 2]);
    //         k_rope_1 = *(half2*)(&local_qkv[HEAD_DIM + HEAD_DIM / 2 + tid * 2]);
    //         cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
    //         sin_reg = {-sin[HEAD_DIM / 2 + tid * 2], -sin[HEAD_DIM / 2 + tid * 2 + 1]};
    //     } else {
    //         q_rope_1 = *(half2*)(&local_qkv[tid * 2 - HEAD_DIM / 2]);
    //         k_rope_1 = *(half2*)(&local_qkv[HEAD_DIM + tid * 2 - HEAD_DIM / 2]);
    //         cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
    //         sin_reg = {sin[tid * 2 - HEAD_DIM / 2], sin[tid * 2 + 1 - HEAD_DIM / 2]};
    //     }
    //     *(half2*)(&local_qkv[tid * 2]) = __hadd2(__hmul2(q_rope, __float22half2_rn(cos_reg)), __hmul2(q_rope_1, __float22half2_rn(sin_reg)));
    //     *(half2*)(&local_qkv[HEAD_DIM + tid * 2]) = __hadd2(__hmul2(k_rope, __float22half2_rn(cos_reg)), __hmul2(k_rope_1, __float22half2_rn(sin_reg)));
    // }

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
    for(int j = 0; j < DIM_BLOCK_REDUCE; j++) {
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
    // DSM Ring All-reduce: local_max
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
    // DSM Ring-All reduce: local_sum
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
        *(uint4*)(&local_qkv[2 * HEAD_DIM + tid * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
    }
    block.sync();

    // DSM Ring-All reduce
    size = HEAD_DIM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[2 * HEAD_DIM]));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    dsm_ring_allreduce<CLUSTER_SIZE, Stage::ATTN>(
        size, tid, HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, &local_qkv[2 * HEAD_DIM], weight);

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
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_3 + j]);
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
        block.sync();
    }
    bar[1].wait(std::move(token[1]));
    tmp = 0.0;
    for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_3) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_3 + j]);
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
    // cluster.sync();

    // // Fused residual and RMSNorm
    // local_sum = 0.0;
    // for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
    //     *(uint4*)(&reg_reduce[0]) = *(uint4*)(&global_reduce[cluster_block_st_id + d]);
    //     *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[d]);
    //     for (int di = 0; di < 8; di++) {
    //         float x = __half2float(reg_input[di]) + __half2float(reg_reduce[di]);
    //         local_sum += x * x;
    //         reg_input[di] = __float2half(x);
    //     }
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
    // if (tid == 0)
    //     cluster_local_sum = local_sum;
    // cluster.sync();
    // // DSM Ring-All reduce
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
    // for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
    //     *(uint4*)(&reg_weight[0]) = *(uint4*)(&w_rms_attn[cluster_block_st_id + d]);
    //     for (int i = 0; i < 8; i++) {
    //         reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
    //     }
    //     *(uint4*)(&input_shmem[d]) = *(uint4*)(&reg_input[0]);
    // }
    // block.sync();

    // // Compute input @ ffn_gate
    // for (int j = 0; j < FFN_DIM_PER_CLUSTER / HEAD_DIM; j++){
    //     tmp_ffn[j] = 0.0;
    // }
    // // Preload weight_gate
    // if (tid == 0) {
    //     cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_gate_up, cluster_head_ffn_idx, cluster_block_st_id, bar[0]);
    //     cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_gate_up_, cluster_head_ffn_idx + TMA_LOAD_ONCE_MAX, cluster_block_st_id, bar[0]);
    //     token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_FFN);
    // } else {
    //     token[0] = bar[0].arrive();
    // }

    // for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
    //     if (tid == 0) {
    //         cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL], &tensor_map_weight_gate_up, cluster_head_ffn_idx, cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
    //         cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_gate_up_, cluster_head_ffn_idx + TMA_LOAD_ONCE_MAX, cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
    //         token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_FFN);
    //     } else {
    //         token[id % 2] = bar[id % 2].arrive();
    //     }
    //     bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
    //     for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
    //         *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
    //         for (int j = 0; j < TMA_LOAD_ONCE_MAX / HEAD_DIM; j++) {
    //             #pragma unroll
    //             for (int d = 0; d < NUM_PER_THREAD; d++) {
    //                 tmp_ffn[j] += __half2float(__hmul(reg_input[d], weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx + i + d) * TMA_LOAD_ONCE_MAX + weight_idx + j * HEAD_DIM]));
    //             }
    //         }
    //         for (int j = 0; j < (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) / HEAD_DIM; j++) {
    //             #pragma unroll
    //             for (int d = 0; d < NUM_PER_THREAD; d++) {
    //                 tmp_ffn[TMA_LOAD_ONCE_MAX / HEAD_DIM + j] += __half2float(__hmul(reg_input[d], weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx + i + d) * (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) + weight_idx + j * HEAD_DIM]));
    //             }
    //         }
    //     }
    // }
    // bar[1].wait(std::move(token[1]));
    // for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
    //     *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
    //     for (int j = 0; j < TMA_LOAD_ONCE_MAX / HEAD_DIM; j++) {
    //         #pragma unroll
    //         for (int d = 0; d < NUM_PER_THREAD; d++) {
    //             tmp_ffn[j] += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx + i + d) * TMA_LOAD_ONCE_MAX + weight_idx + j * HEAD_DIM]));
    //         }
    //     }
    //     for (int j = 0; j < (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) / HEAD_DIM; j++) {
    //         #pragma unroll
    //         for (int d = 0; d < NUM_PER_THREAD; d++) {
    //             tmp_ffn[TMA_LOAD_ONCE_MAX / HEAD_DIM + j] += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx + i + d) * (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) + weight_idx + j * HEAD_DIM]));
    //         }
    //     }
    // }
    // for (int j = 0; j < FFN_DIM_PER_CLUSTER / HEAD_DIM; j++){
    //     local_qkv[j * HEAD_DIM + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp_ffn[j]);
    // }

    // // Compute input @ ffn_up
    // for (int j = 0; j < FFN_DIM_PER_CLUSTER / HEAD_DIM; j++){
    //     tmp_ffn[j] = 0.0;
    // }
    // // Preload weight_up
    // if (tid == 0) {
    //     cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_gate_up, cluster_head_ffn_idx, HIDDEN_DIM + cluster_block_st_id, bar[0]);
    //     cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_gate_up_, cluster_head_ffn_idx + TMA_LOAD_ONCE_MAX, HIDDEN_DIM + cluster_block_st_id, bar[0]);
    //     token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_FFN);
    // } else {
    //     token[0] = bar[0].arrive();
    // }

    // for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
    //     if (tid == 0) {
    //         cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL], &tensor_map_weight_gate_up, cluster_head_ffn_idx, HIDDEN_DIM + cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
    //         cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_gate_up_, cluster_head_ffn_idx + TMA_LOAD_ONCE_MAX, HIDDEN_DIM + cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
    //         token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_FFN);
    //     } else {
    //         token[id % 2] = bar[id % 2].arrive();
    //     }
    //     bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
    //     for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
    //         *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
    //         for (int j = 0; j < TMA_LOAD_ONCE_MAX / HEAD_DIM; j++) {
    //             #pragma unroll
    //             for (int d = 0; d < NUM_PER_THREAD; d++) {
    //                 tmp_ffn[j] += __half2float(__hmul(reg_input[d], weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx + i + d) * TMA_LOAD_ONCE_MAX + weight_idx + j * HEAD_DIM]));
    //             }
    //         }
    //         for (int j = 0; j < (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) / HEAD_DIM; j++) {
    //             #pragma unroll
    //             for (int d = 0; d < NUM_PER_THREAD; d++) {
    //                 tmp_ffn[TMA_LOAD_ONCE_MAX / HEAD_DIM + j] += __half2float(__hmul(reg_input[d], weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx + i + d) * (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) + weight_idx + j * HEAD_DIM]));
    //             }
    //         }
    //     }
    // }
    // bar[1].wait(std::move(token[1]));
    // for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
    //     *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
    //     for (int j = 0; j < TMA_LOAD_ONCE_MAX / HEAD_DIM; j++) {
    //         #pragma unroll
    //         for (int d = 0; d < NUM_PER_THREAD; d++) {
    //             tmp_ffn[j] += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx + i + d) * TMA_LOAD_ONCE_MAX + weight_idx + j * HEAD_DIM]));
    //         }
    //     }
    //     for (int j = 0; j < (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) / HEAD_DIM; j++) {
    //         #pragma unroll
    //         for (int d = 0; d < NUM_PER_THREAD; d++) {
    //             tmp_ffn[TMA_LOAD_ONCE_MAX / HEAD_DIM + j] += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx + i + d) * (FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX) + weight_idx + j * HEAD_DIM]));
    //         }
    //     }
    // }
    // for (int j = 0; j < FFN_DIM_PER_CLUSTER / HEAD_DIM; j++){
    //     local_qkv[MAX_SMEM_DIM + j * HEAD_DIM + warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp_ffn[j]);
    // }
    // block.sync();

    // // DSM Ring-All reduce
    // size = FFN_DIM_PER_CLUSTER * 2 * sizeof(half);
    // src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    // dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    // dsm_ring_allreduce<CLUSTER_SIZE, Stage::FFN>(
    //     size, tid, HEAD_DIM, cluster_block_id,  
    //     src_addr, dst_addr, bar_ptr, 
    //     neighbor_dst_bar, local_qkv, weight);
    
    // // Compute up_gate mul and down_proj
    // if (tid == 0) {
    //     cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_down, cluster_block_st_id, cluster_head_ffn_idx, bar[0]);
    //     cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_down_, cluster_block_st_id, cluster_head_ffn_idx + TMA_LOAD_ONCE_MAX, bar[0]);
    //     token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_FFN);
    // } else {
    //     token[0] = bar[0].arrive();
    // }

    // for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
    //     if (tid == 0) {
    //         cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL], &tensor_map_weight_down, cluster_block_st_id + id * TMA_LOAD_ONCE, cluster_head_ffn_idx, bar[id % 2]);
    //         cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN], &tensor_map_weight_down_, cluster_block_st_id + id * TMA_LOAD_ONCE, cluster_head_ffn_idx + TMA_LOAD_ONCE_MAX, bar[id % 2]);
    //         token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_FFN);
    //     } else {
    //         token[id % 2] = bar[id % 2].arrive();
    //     }
    //     bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
    //     tmp = 0.0;
    //     for (int j = 0; j < TMA_LOAD_ONCE_MAX; j+=NUM_PER_ROW_3) {
    //         *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3 + j]);
    //         *(uint4*)(&reg_reduce[0]) = *(uint4*)(&local_qkv[MAX_SMEM_DIM + input_idx_3 + j]);
    //         #pragma unroll
    //         for (int d = 0; d < NUM_PER_THREAD; d++) {
    //             tmp += __half2float(__hmul(__hmul(reg_input[d], reg_reduce[d]), weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]));
    //         }
    //     }
    //     for (int j = 0; j < FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX; j+=NUM_PER_ROW_3) {
    //         *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3 + TMA_LOAD_ONCE_MAX + j]);
    //         *(uint4*)(&reg_reduce[0]) = *(uint4*)(&local_qkv[MAX_SMEM_DIM + input_idx_3 + TMA_LOAD_ONCE_MAX + j]);
    //         #pragma unroll
    //         for (int d = 0; d < NUM_PER_THREAD; d++) {
    //             tmp += __half2float(__hmul(__hmul(reg_input[d], reg_reduce[d]), weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]));
    //         }
    //     }
    //     #pragma unroll
    //     for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
    //         tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    //     }
    //     if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
    //         atomicAdd(&output[cluster_block_st_id + weight_idx_3 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    //     }
    // }
    // bar[1].wait(std::move(token[1]));
    // tmp = 0.0;
    // for (int j = 0; j < TMA_LOAD_ONCE_MAX; j+=NUM_PER_ROW_3) {
    //     *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3 + j]);
    //     *(uint4*)(&reg_reduce[0]) = *(uint4*)(&local_qkv[MAX_SMEM_DIM + input_idx_3 + j]);
    //     #pragma unroll
    //     for (int d = 0; d < NUM_PER_THREAD; d++) {
    //         tmp += __half2float(__hmul(__hmul(reg_input[d], reg_reduce[d]), weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]));
    //     }
    // }
    // for (int j = 0; j < FFN_DIM_PER_CLUSTER - TMA_LOAD_ONCE_MAX; j+=NUM_PER_ROW_3) {
    //     *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_3 + TMA_LOAD_ONCE_MAX + j]);
    //     *(uint4*)(&reg_reduce[0]) = *(uint4*)(&local_qkv[MAX_SMEM_DIM + input_idx_3 + TMA_LOAD_ONCE_MAX + j]);
    //     #pragma unroll
    //     for (int d = 0; d < NUM_PER_THREAD; d++) {
    //         tmp += __half2float(__hmul(__hmul(reg_input[d], reg_reduce[d]), weight[TMA_LOAD_ONCE_NUM_FFN_TOTAL + TMA_LOAD_ONCE_NUM_FFN + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]));
    //     }
    // }
    // #pragma unroll
    // for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
    //     tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    // }
    // if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
    //     atomicAdd(&output[cluster_block_st_id + weight_idx_3 + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    // }
}