#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <math_constants.h> 
#include "../../dsm.cuh"
#include "config.h"
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

// wrapper of cp.async
__device__ __forceinline__ void cp_async_pred_load_128b(half* smem_ptr, const half* gmem_ptr, bool predicate) {
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int src_in_bytes = predicate ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
}

// wrapper of cp.async.commit_group
__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// wrapper of cp.async.wait_group
template <size_t n>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

  __device__ __forceinline__ size_t protective_get_kv_offset(int start_idx, int end_idx, int offset) {
    if (start_idx + offset < end_idx) {
        return start_idx + offset;
    } else {
        return 0;
    }
  }

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) LlamaDecoderLayerBatchDecodeWithPagedKVCacheKernel(
    half* output,        // batch_size * hidden_dim
    half* input,         // batch_size * hidden_dim
    half* residual,      // batch_size * hidden_dim
    half* residual_output,
    half* w_rms_input,   // hidden_dim
    float eps,
    int64_t* positions,
    float* cos_sin,
    uint64_t* k_cache_ptrs,
    uint64_t* v_cache_ptrs,
    int layer_id,
    int* paged_kv_indptr,
    int* paged_kv_indices,
    const __grid_constant__ CUtensorMap tensor_map, // 3 * hidden_dim * hidden_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_o // hidden_dim * hidden_dim
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t batch_id         = grid.cluster_rank() / HEAD_NUM;
    const uint32_t head_id          = grid.cluster_rank() % HEAD_NUM;
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; 
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t tile_row = tid / NUM_THREAD_PER_ROW_2;
    const uint32_t tile_col = tid % NUM_THREAD_PER_ROW_2;

    // Init shared memory
    extern __shared__ uint8_t shmem_base[];
    half* weight = reinterpret_cast<half*>((uintptr_t)(shmem_base) + 127 & ~127);
    half* local_qkv = weight + 2 * TMA_LOAD_ONCE * MAX_SMEM_DIM;
    half* input_shmem = local_qkv + 3 * HEAD_DIM;
    float* reduction = reinterpret_cast<float*>(input_shmem + DIM_PER_BLOCK);

    __shared__ float cluster_local_sum, cluster_local_max;

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

    // Init registers
    float local_sum = 0.0, rms_rcp = 0.0, tmp = 0.0, local_max = -CUDART_INF_F, pre_max = -CUDART_INF_F, scale = 0.0, softmax_scale = __frsqrt_rn(HEAD_DIM) * 1.44269504088896340736f;
    half __align__(16) reg_input[NUM_PER_THREAD], reg_residual[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
    float reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    // half2 q_rope, q_rope_1, k_rope, k_rope_1;
    // float2 cos_reg, sin_reg;
    float q_rope, q_rope_1, k_rope, k_rope_1, cos_reg, sin_reg;
    uint32_t size;
    uint32_t src_addr, dst_addr, neighbor_dst_bar = 0;
    float __align__(16) qk[DEC_TILE];

    // Precompute some indices
    uint input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    uint input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    uint weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
    uint input_idx_3 = (lane_id % NUM_THREAD_PER_ROW_3) * NUM_PER_THREAD;
    uint weight_idx_3 = warp_id * NUM_ROW_PER_WARP_3 + lane_id / NUM_THREAD_PER_ROW_3;
    uint cluster_block_st_id = cluster_block_id * DIM_PER_BLOCK;
    uint cluster_head_idx = head_id * HEAD_DIM;
    half* k_cache = reinterpret_cast<half*>(k_cache_ptrs[layer_id]);
    half* v_cache = reinterpret_cast<half*>(v_cache_ptrs[layer_id]);
    int start_idx = paged_kv_indptr[batch_id];
    int end_idx = paged_kv_indptr[batch_id + 1] - 1;
    const uint32_t SEQ_LEN = end_idx - start_idx;
    const uint32_t KV_DIM_PER_BLOCK = ((SEQ_LEN + CLUSTER_SIZE - 1) / CLUSTER_SIZE + (TMA_LOAD_ONCE_ATTN - 1)) & ~(TMA_LOAD_ONCE_ATTN - 1);

    // RMSNorm
    for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[batch_id * HIDDEN_DIM + cluster_block_st_id + d]);
        *(uint4*)(&reg_residual[0]) = *(uint4*)(&residual[batch_id * HIDDEN_DIM + cluster_block_st_id + d]);
        for (int di = 0; di < 8; di++) {
            reg_input[di] = __float2half(__half2float(reg_input[di]) + __half2float(reg_residual[di]));
            local_sum += __half2float(reg_input[di]) * __half2float(reg_input[di]);
        }
        *(uint4*)(&residual_output[batch_id * HIDDEN_DIM + cluster_block_st_id + d]) = *(uint4*)(&reg_input[0]);
    }
    block.sync();
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
    // ClusterReduce
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
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_block_st_id, cluster_head_idx, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_block_st_id + id * TMA_LOAD_ONCE, cluster_head_idx, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
        }
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
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
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_block_st_id, cluster_head_idx + HEAD_NUM * HEAD_DIM, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_block_st_id + id * TMA_LOAD_ONCE, cluster_head_idx + HEAD_NUM * HEAD_DIM, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
        }
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
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
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_block_st_id, cluster_head_idx + 2 * HEAD_NUM * HEAD_DIM, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_block_st_id + id * TMA_LOAD_ONCE, cluster_head_idx + 2 * HEAD_NUM * HEAD_DIM, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
        }
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
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

    // ClusterReduce
    size = (HEAD_DIM * 3) * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    cluster_reduce<CLUSTER_SIZE, Stage::LINEAR>(
        size, tid, HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, weight);

    // RoPE
    q_rope = __half2float(local_qkv[tid]);
    k_rope = __half2float(local_qkv[HEAD_DIM + tid]);
    cos_reg = cos_sin[positions[batch_id] * HEAD_DIM + tid % (HEAD_DIM / 2)];
    sin_reg = cos_sin[positions[batch_id] * HEAD_DIM + tid % (HEAD_DIM / 2) + HEAD_DIM / 2];
    if (tid < HEAD_DIM / 2) {
        q_rope_1 = __half2float(local_qkv[HEAD_DIM / 2 + tid]);
        k_rope_1 = __half2float(local_qkv[HEAD_DIM + HEAD_DIM / 2 + tid]);
    } else {
        q_rope_1 = __half2float(local_qkv[tid - HEAD_DIM / 2]);
        k_rope_1 = __half2float(local_qkv[HEAD_DIM + tid - HEAD_DIM / 2]);
    }
    block.sync();
    if (tid < HEAD_DIM / 2) {
        local_qkv[tid] = __float2half(q_rope * cos_reg - q_rope_1 * sin_reg);
        local_qkv[HEAD_DIM + tid] = __float2half(k_rope * cos_reg - k_rope_1 * sin_reg);
    } else {
        local_qkv[tid] = __float2half(q_rope * cos_reg + q_rope_1 * sin_reg);
        local_qkv[HEAD_DIM + tid] = __float2half(k_rope * cos_reg + k_rope_1 * sin_reg);
    }

    // Save new kv
    cluster.sync();
    if (cluster_block_id == 0) {
        k_cache[paged_kv_indices[end_idx] * HIDDEN_DIM + head_id * HEAD_DIM + tid] = local_qkv[HEAD_DIM + tid];
        v_cache[paged_kv_indices[end_idx] * HIDDEN_DIM + head_id * HEAD_DIM + tid] = local_qkv[2 * HEAD_DIM + tid];
    }
    cluster.sync();

    // Compute flash-decoding
    local_sum = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2]);
    block.sync();

    // Preload kv_cache
    for (int j = 0; j < DEC_TILE; j++) {
        cp_async_pred_load_128b(
            &weight[0 + (weight_idx_2 + j) * HEAD_DIM + input_idx_2], 
            &k_cache[(paged_kv_indices[protective_get_kv_offset(start_idx, end_idx, cluster_block_id * KV_DIM_PER_BLOCK + weight_idx_2 + j)]) * HIDDEN_DIM + cluster_head_idx + input_idx_2],
            (start_idx + cluster_block_id * KV_DIM_PER_BLOCK + weight_idx_2 + j) < end_idx
        );
    }
    cp_async_commit_group();
    for (int j = 0; j < DEC_TILE; j++) {
        cp_async_pred_load_128b(
            &weight[TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2], 
            &v_cache[(paged_kv_indices[protective_get_kv_offset(start_idx, end_idx, cluster_block_id * KV_DIM_PER_BLOCK + weight_idx_2 + j)]) * HIDDEN_DIM + cluster_head_idx + input_idx_2],
            (start_idx + cluster_block_id * KV_DIM_PER_BLOCK + weight_idx_2 + j) < end_idx
        );
    }
    cp_async_commit_group();

    // mainloop
    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        for (int j = 0; j < DEC_TILE; j++) {
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2], 
                &k_cache[(paged_kv_indices[protective_get_kv_offset(start_idx, end_idx, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j)]) * HIDDEN_DIM + cluster_head_idx + input_idx_2],
                (start_idx + cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j) < end_idx
            );
        }
        cp_async_commit_group();
        cp_async_wait_group<2>();
        block.sync();

        pre_max = local_max;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            qk[j] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                // qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
                qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
            }
            qk[j] = qk[j] * softmax_scale;
            local_max = max(local_max, qk[j]);
        }
        scale = ptx_exp2(pre_max - local_max);
        local_sum *= scale;
        // For filled zeros
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            if ((KV_DIM_PER_BLOCK * cluster_block_id + (id - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j) < SEQ_LEN) {
                qk[j] = ptx_exp2(qk[j] - local_max);
                local_sum += qk[j];
            }
        }
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++) {
            // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
            reg_reduce[j] = reg_reduce[j] * scale;
        }
        for (int j = 0; j < DEC_TILE; j++) {
            cp_async_pred_load_128b(
                &weight[(id % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2], 
                &v_cache[(paged_kv_indices[protective_get_kv_offset(start_idx, end_idx, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j)]) * HIDDEN_DIM + cluster_head_idx + input_idx_2],
                (start_idx + cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j) < end_idx
            );
        }
        cp_async_commit_group();
        cp_async_wait_group<2>();
        block.sync();
        for (int j = 0; j < DEC_TILE; j++) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                // reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
                reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
            }
        }
    }
    // end: mainloop

    // epilogue
    cp_async_wait_group<1>();
    block.sync();

    pre_max = local_max;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        qk[j] = 0.0f;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
            qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
        }
        qk[j] = qk[j] * softmax_scale;
        local_max = max(local_max, qk[j]);
    }
    scale = ptx_exp2(pre_max - local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        if ((KV_DIM_PER_BLOCK * cluster_block_id + (KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j) < SEQ_LEN) {
            qk[j] = ptx_exp2(qk[j] - local_max);
            local_sum += qk[j];
        }
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        reg_reduce[j] = reg_reduce[j] * scale;
    }

    cp_async_wait_group<0>();
    block.sync();

    for (int j = 0; j < DEC_TILE; j++) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
            reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
        }
    }
    block.sync();

    // Process KV of current token
    if (cluster_block_id == 0 && warp_id == 0) {
        if (lane_id / NUM_THREAD_PER_ROW_2 == 1) {
            pre_max = local_max;
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[HEAD_DIM + input_idx_2]); 
            qk[0] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                // qk[j] += __half2float(__hmul(reg_input[d], reg_weight[d]));
                qk[0] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[0] += __shfl_xor_sync(0xffffffff, qk[0], mask);
        }
        if (lane_id / NUM_THREAD_PER_ROW_2 == 1) {
            qk[0] = qk[0] * softmax_scale;
            local_max = max(local_max, qk[0]); 
            scale = ptx_exp2(pre_max - local_max);
            local_sum *= scale;
            qk[0] = ptx_exp2(qk[0] - local_max);
            local_sum += qk[0];
            #pragma unroll
            for (int j = 0; j < NUM_PER_THREAD; j++) {
                // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
                reg_reduce[j] = reg_reduce[j] * scale;
            }
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_2]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                // reg_reduce[d] = __hadd(reg_reduce[d], __float2half(qk[j] * __half2float(reg_weight[d])));
                reg_reduce[d] = reg_reduce[d] + qk[0] * __half2float(reg_weight[d]);
            }
        }
    }
    block.sync();

    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
    }
    // *(uint4*)(&weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        reduction[tile_row * 2] = local_max;
        reduction[tile_row * 2 + 1] = local_sum;
    }
    block.sync();
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    local_sum = 0.0, local_max = 0.0;
    #pragma unroll
    for(int j = 0; j < DIM_BLOCK_REDUCE / 2; j++) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * HEAD_DIM + tile_col * NUM_PER_THREAD]);
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = max(m, local_max);
        scale = ptx_exp2(m - local_max);
        s *= scale;
        local_sum = local_sum * ptx_exp2(pre_max - local_max) + s;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // reg_reduce[d] = __hadd(__hmul(reg_reduce[d], __float2half(ptx_exp2(pre_max - local_max))), __hmul(reg_input[d], __float2half(scale)));
            reg_reduce[d] = reg_reduce[d] * ptx_exp2(pre_max - local_max) + __half2float(reg_input[d]) * scale;
        }
    }
    block.sync();

    pre_max = local_max;
    if(tid == 0) {
        cluster_local_max = local_max;
    }
    cluster.sync();
    // ClusterReduce: local_max
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
    scale = ptx_exp2(pre_max - cluster_local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(scale));
        reg_reduce[j] = reg_reduce[j] * scale;
    }
    if(tid == 0) {
        cluster_local_sum = local_sum;
    }
    cluster.sync();
    // ClusterReduce: local_sum
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
        // reg_reduce[j] = __hmul(reg_reduce[j], __float2half(__frcp_rn(cluster_local_sum)));
        reg_reduce[j] = reg_reduce[j] * __frcp_rn(cluster_local_sum);
    }
    if(tid < NUM_THREAD_PER_ROW_2) {
        // NOTE: revised vectorized store
        // *(uint4*)(&local_qkv[2 * HEAD_DIM + tid * NUM_PER_THREAD]) = *(uint4*)(&reg_reduce[0]);
        #pragma unroll
        for (int i = 0; i < NUM_PER_THREAD; i++) {
            local_qkv[2 * HEAD_DIM + tid * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
        }
    }
    block.sync();

    // ClusterReduce
    size = HEAD_DIM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[2 * HEAD_DIM]));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    cluster_reduce<CLUSTER_SIZE, Stage::ATTN>(
        size, tid, HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, &local_qkv[2 * HEAD_DIM], weight);

    // Set output buffer to zero
    *(uint4*)(&output[batch_id * HIDDEN_DIM + cluster_block_st_id + tid * NUM_PER_THREAD]) = {0, 0, 0, 0};
    block.sync();

    // Compute output @ w_o
    // Preload w_o
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_o, cluster_head_idx, cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_o, cluster_head_idx, cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        tmp = 0.0;
        for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_3) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_3 + j]);
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM + weight_idx_3 * HEAD_DIM + input_idx_3 + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
        if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
            atomicAdd(&output[batch_id * HIDDEN_DIM + cluster_block_st_id + weight_idx_3 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
            // input_shmem[weight_idx_3 + (id - 1) * TMA_LOAD_ONCE] = __float2half(tmp);
        }
        block.sync();
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    tmp = 0.0;
    for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_3) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_3 + j]);
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2 * TMA_LOAD_ONCE_NUM + weight_idx_3 * HEAD_DIM + input_idx_3 + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            // tmp += __half2float(__hmul(reg_input[d], weight[TMA_LOAD_ONCE_NUM + (input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]));
            tmp += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
        atomicAdd(&output[batch_id * HIDDEN_DIM + cluster_block_st_id + weight_idx_3 + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
        // input_shmem[weight_idx_3 + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE] = __float2half(tmp);
    }
    // block.sync();
    // for (int i = 0; i < DIM_PER_BLOCK / BLOCK_SIZE; i++) {
        // atomicAdd(&output[cluster_block_st_id + tid * DIM_PER_BLOCK / BLOCK_SIZE + i], input_shmem[tid * DIM_PER_BLOCK / BLOCK_SIZE + i]);
    // }
}