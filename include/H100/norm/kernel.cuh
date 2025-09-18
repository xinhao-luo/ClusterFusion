#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include "config.h"
namespace cg = cooperative_groups;


__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) RMSNormKernel(
    half* output, // 1 * hidden_dim
    half* input,  // 1 * hidden_dim
    half* weight// hidden_dim
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t batch_id         = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; 
    const uint32_t warp_id = tid / WARP_SIZE;

    // Init shared memory
    __shared__ float reduction[NUM_WARPS];
    __shared__ float cluster_local_sum;

    // Init registers
    float local_sum = 0.0, eps = 1e-6, rms_rcp = 0.0;
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
    float* dst_shmem;

    // Precompute some indices
    uint cluster_block_st_idx = cluster_block_id * DIM_PER_BLOCK;

    // RMSNorm
    for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[batch_id * HIDDEN_DIM + cluster_block_st_idx + d]);
        #pragma unroll
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
    block.sync();
    // ClusterReduce
    if (tid == 0) {
        int dst_cta = (cluster_block_id + 1) % cluster.num_blocks();
        dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);  
    }
    cluster.sync();
    if (tid == 0) {
        atomicAdd(dst_shmem, local_sum);
    }
    cluster.sync();
    rms_rcp = __frsqrt_rn(cluster_local_sum / HIDDEN_DIM + eps);
    for (int d = tid * 8; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[cluster_block_st_idx + d]);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
        }
        *(uint4*)(&output[batch_id * HIDDEN_DIM + cluster_block_st_idx + d]) = *(uint4*)(&reg_input[0]);
    }
}