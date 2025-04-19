#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <cuda.h>    
#include <iostream>
#include <random>
namespace cg = cooperative_groups;

// nvcc --generate-code=arch=compute_90a,code=sm_90a -O3 -std=c++17 -lcuda norm.cu -o test && ./test

#define BATCH_SIZE 64
#define HIDDEN_DIM 8192   
#define CLUSTER_SIZE 2 // 2 4 8 16
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)
#define NUM_PER_THREAD 8
#define BLOCK_SIZE (DIM_PER_BLOCK / NUM_PER_THREAD)
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 5.0);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(0.1f);
        }   
    }   
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) RMSNormKernel(
    half* output, // 1 * hidden_dim
    half* input,  // 1 * hidden_dim
    half* w_rms_input// hidden_dim
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
    // DSM Ring All-reduce
    for (int i = 1; i < cluster.num_blocks(); i++) {
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
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&w_rms_input[cluster_block_st_idx + d]);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            reg_input[i] = __float2half(__half2float(reg_input[i]) * rms_rcp * __half2float(reg_weight[i]));
        }
        *(uint4*)(&output[batch_id * HIDDEN_DIM + cluster_block_st_idx + d]) = *(uint4*)(&reg_input[0]);
    }
}

int main(int argc, char** argv) {
    half *h_input, *d_input;
    half *h_rms_input, *d_rms_input;
    h_input = new half[BATCH_SIZE * HIDDEN_DIM];
    h_rms_input = new half[HIDDEN_DIM];

    fill_matrix(h_input, BATCH_SIZE * HIDDEN_DIM);
    fill_matrix(h_rms_input, HIDDEN_DIM);

    cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * BATCH_SIZE * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_input), sizeof(half) * HIDDEN_DIM);
    
    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * BATCH_SIZE * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_input), h_rms_input, sizeof(half) * HIDDEN_DIM, cudaMemcpyHostToDevice);

    half* h_output, *d_output;
    h_output = new half[BATCH_SIZE * HIDDEN_DIM];
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * BATCH_SIZE * HIDDEN_DIM);

    cudaDeviceSynchronize();

    dim3 grid(BATCH_SIZE * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    int wmup = 100;
    int test = 1000;
    for (int i = 0; i < wmup; i++) {
        RMSNormKernel<<<grid, block>>>(
            d_output,
            d_input,
            d_rms_input
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
        RMSNormKernel<<<grid, block>>>(
            d_output,
            d_input,
            d_rms_input
        );
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / test * 1e3 << " us" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * BATCH_SIZE * HIDDEN_DIM, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < BATCH_SIZE * HIDDEN_DIM; i++)
    //     printf("%f, ", __half2float(h_output[i]));
    return 0;
}