#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <math_constants.h> 
#include "../../dsm.cuh"
#include "config.h"
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

// #define DEBUG
#ifdef DEBUG
#define PRINT_HEAD 1
#endif

// Neox-style RoPE for sglang.
// If commented, we will use GPT-J style RoPE for tests/models/llama.py
#define NEOX_STYLE_ROPE

__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) LlamaDecoderLayerKernel(
    half* output, // 1 * hidden_dim
    half* normed_output,
    half* k_output,
    half* v_output,
    half* input,  // 1 * hidden_dim
    half* residual,  // 1 * hidden_dim
    half* w_rms_input,// hidden_dim
    float eps,
    float* cos,       // head_dim
    float* sin,       // head_dim
    half* k_cache,
    half* v_cache,
    const __grid_constant__ CUtensorMap tensor_map, // 3 * hidden_dim * hidden_dim
    const __grid_constant__ CUtensorMap tensor_map_k_cache, // seqlen * head_num * head_dim
    const __grid_constant__ CUtensorMap tensor_map_v_cache, // seqlen * head_num * head_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_o, // hidden_dim * hidden_dim
    const uint32_t SEQ_LEN,
    const uint32_t KV_DIM_PER_BLOCK
)

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
LlamaDecoderLayerKernelPaged(
    half* __restrict__ output,              // [1, hidden_dim]
    const half* __restrict__ normed_output, // [1, hidden_dim]
    half* __restrict__ k_output,            // [1, head_num, head_dim]
    half* __restrict__ v_output,            // [1, head_num, head_dim]
    const half* __restrict__ input,         // [1, hidden_dim]
    const half* __restrict__ residual,      // [1, hidden_dim]
    const half* __restrict__ w_rms_input,   // [hidden_dim]
    float eps,
    const float* __restrict__ cos,          // [head_dim]
    const float* __restrict__ sin,          // [head_dim]

    const half* __restrict__ k_cache_pages, // [max_pages, page_size, num_kv_heads, head_dim]
    const half* __restrict__ v_cache_pages, // [max_pages, page_size, num_kv_heads, head_dim]

    const uint32_t* __restrict__ indptr,     // [batch_size+1]
    const uint32_t* __restrict__ indices,    // [total_pages_used]
    const uint32_t* __restrict__ last_page_len, // [batch_size]

    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t page_size,
    uint32_t max_seq_len,
    uint32_t window_left,
    uint32_t kv_layout, 
    float   logits_soft_cap,

    uint64_t stride_pages_p,
    uint64_t stride_pages_h,
    uint64_t stride_pages_d,

    const uint32_t SEQ_LEN,
    const uint32_t KV_DIM_PER_BLOCK
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
#ifdef DEBUG
    if (tid == 0 && head_id == PRINT_HEAD && cluster_block_id == 2) {
        printf("PRINT_HEAD: %d\n", PRINT_HEAD);
    }
#endif

    // Init shared memory
    // __shared__ __align__(16) half input_shmem[DIM_PER_BLOCK];
    // __shared__ float reduction[2 * DIM_BLOCK_REDUCE];
    // __shared__ alignas(128) half weight[2 * TMA_LOAD_ONCE * MAX_SMEM_DIM];
    // __shared__ __align__(16) half local_qkv[MAX_SMEM_DIM + MAX_SMEM_DIM + HEAD_DIM];
    extern __shared__ uint8_t shmem_base[];
    half* weight = reinterpret_cast<half*>((uintptr_t)(shmem_base) + 127 & ~127);
    half* local_qkv = weight + 2 * TMA_LOAD_ONCE * MAX_SMEM_DIM;
    half* input_shmem = local_qkv + 3 * HEAD_DIM;
    float* reduction = reinterpret_cast<float*>(input_shmem + DIM_PER_BLOCK);
    // half* input_shmem = reinterpret_cast<half*>(shmem_base);
    // float* reduction  = reinterpret_cast<float*>(shmem_base + DIM_PER_BLOCK * sizeof(half));
    // half* weight      = reinterpret_cast<half*>((uintptr_t)(shmem_base + DIM_PER_BLOCK * sizeof(half) + 2 * DIM_BLOCK_REDUCE * sizeof(float)) + 127 & ~127);
    // half* local_qkv   = reinterpret_cast<half*>((uintptr_t)(weight + 2 * TMA_LOAD_ONCE * MAX_SMEM_DIM) + 127 & ~127);

    __shared__ float cluster_local_sum, cluster_local_max;

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

    // RMSNorm
    for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[cluster_block_st_id + d]);
        *(uint4*)(&reg_residual[0]) = *(uint4*)(&residual[cluster_block_st_id + d]);
        for (int di = 0; di < 8; di++) {
            reg_input[di] = __float2half(__half2float(reg_input[di]) + __half2float(reg_residual[di]));
            local_sum += __half2float(reg_input[di]) * __half2float(reg_input[di]);
        }
        *(uint4*)(&residual[cluster_block_st_id + d]) = *(uint4*)(&reg_input[0]);
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

    // tmp: output normed
    if (head_id == 0) {
        *(uint4*)(&normed_output[cluster_block_st_id + tid * 8]) = *(uint4*)(&input_shmem[tid * 8]);
    }
}
