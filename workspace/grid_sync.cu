#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <cuda.h>    
#include <iostream>
#include <random>
#include <stdio.h>

// nvcc --generate-code=arch=compute_90a,code=sm_90a -O3 -std=c++17 -lcuda grid_sync.cu -o test && ./test

// #define HEAD_NUM 32
// #define HEAD_DIM 128    
// #define HIDDEN_DIM (HEAD_NUM * HEAD_DIM) 

// #define NUM_WARPS 4 // 4 8 16 32
// #define WARP_SIZE 32
// #define BLOCK_SIZE (NUM_WARPS * WARP_SIZE) 
// #define CLUSTER_SIZE 4 // 2 4 8 16
// #define GRID_SIZE (HEAD_NUM * CLUSTER_SIZE)
// #define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)

// template <typename T>
// void fill_matrix(T* mat, int sz) {
//     std::random_device r;
//     std::mt19937 rng(r());
//     std::normal_distribution<float> norm_dist(0.0, 5.0);
//     for (int i = 0; i < sz; i++) {
//         if constexpr(std::is_same<T, half>::value) {
//             mat[i] = __float2half(0.01f);
//         }   
//     }   
// }

__global__ void single_decode()
{
    namespace cg = cooperative_groups;
    cg::grid_group grid             = cg::this_grid();
    // cg::thread_block block          = cg::this_thread_block();
    // cg::cluster_group cluster       = cg::this_cluster();
    // const uint32_t head_id          = grid.cluster_rank();
    // const uint32_t cluster_block_id = cluster.block_rank();
    // const uint32_t tid              = block.thread_rank();
    // const uint32_t lane_id = tid % WARP_SIZE; 
    // const uint32_t warp_id = tid / WARP_SIZE;

    // // Precompute some indices
    // uint cluster_st_id = head_id * HEAD_DIM;

    // //
    // __shared__ __align__(16) half input_shmem[HEAD_DIM];
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("1234 \n");
    grid.sync();
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("123456768 \n");
    // Load input to shared memory
    // #pragma unroll
    // for (int i = tid; i < HEAD_DIM; i+=BLOCK_SIZE) {
    //     input_shmem[i] = input[cluster_st_id + i];
    // }
    // block.sync();

    // //
    // atomicAdd(&output[cluster_st_id + tid], input_shmem[tid]);
    // // grid.sync();
    // #pragma unroll
    // for (int i = tid; i < HEAD_DIM; i+=BLOCK_SIZE) {
    //     input_shmem[i] = output[cluster_st_id + i];
    // }
    // block.sync();

    // atomicAdd(&output[cluster_st_id + tid], input_shmem[tid]);
    // grid.sync();
}

int main() {
    // half *h_input, *d_input;
    // half* h_output, *d_output;
    // h_input = new half[1 * HIDDEN_DIM];
    // h_output = new half[1 * HIDDEN_DIM];

    // fill_matrix(h_input, 1 * HIDDEN_DIM);

    // cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * 1 * HIDDEN_DIM);
    // cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * 1 * HIDDEN_DIM);
    // cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyHostToDevice);
    
    cudaLaunchConfig_t config = {0};
    config.gridDim = 8;
    config.blockDim = 128;
    config.dynamicSmemBytes = 0;
    cudaFuncSetAttribute((void *)single_decode, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute((void *)single_decode, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 4; 
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    // launch
    // void *kernelArgs[] = {&d_input, &d_output};
    // dim3 dimBlock(BLOCK_SIZE);
    // dim3 dimGrid(HEAD_NUM, CLUSTER_SIZE);
    int wmup = 1;
    int test = 0;
    for (int i = 0; i < wmup; i++) {
        cudaLaunchKernelEx(&config, single_decode);
        // cudaLaunchCooperativeKernel((void*)single_decode, dimGrid, dimBlock, kernelArgs);
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
        cudaLaunchKernelEx(&config, single_decode);
        // cudaLaunchCooperativeKernel((void*)single_decode, dimGrid, dimBlock, kernelArgs);
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / test * 1e3 << " us" << std::endl;
    // cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < HIDDEN_DIM; i++)
    //     printf("%f, ", __half2float(h_output[i]));
    // printf("\n");
    return 0;
}