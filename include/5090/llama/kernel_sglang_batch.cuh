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
}
