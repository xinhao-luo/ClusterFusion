// CUDA_VISIBLE_DEVICES=4 nvcc reduce_dsm.cu -o test -arch=sm_90 --std=c++17 && ./test                                           
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cooperative_groups.h>


const int cluster_size = 2; // 2 4 8 16
const int num_blocks = 4096;
const int num_threads = 512; // 64 128 256 512
const int data = 4096 * 4;
const int repeats = 4;

template<int ClusterSize>
struct __align__(16) SharedStorage {
    alignas(16) float smem_q[data];
};

template<int ClusterSize>
__global__ void clusterHist_kernel(float *global_output) {

    namespace cg = cooperative_groups;

    cg::cluster_group cluster = cg::this_cluster();

    const int clusterBlockRank = cluster.block_rank();

    extern __shared__ float smem[];

    using SharedStorageType = SharedStorage<ClusterSize>;
    SharedStorageType* shared_storage = reinterpret_cast<SharedStorageType*>(smem);

    float* smem_q = shared_storage->smem_q;

    // for (int j = threadIdx.x; j < num_threads; j += num_threads) {
    //     smem_q[j] = 1.0;
    // }
    // cluster.sync();
    /*
    float *dst_smem = 0;
    float tmp;
    dst_smem = cluster.map_shared_rank(smem_q, 0);
    for (int t = 0; t < repeats; t++) {
        if(clusterBlockRank != 0) {
            for (int j = threadIdx.x; j < data ; j += num_threads) {
                atomicAdd(&dst_smem[j], smem_q[j]);
            }
        }
    }
    */
    if (clusterBlockRank == 0) {
        for (int t = 0; t < repeats; t++) {
            for (int b = 1; b < ClusterSize; b++) {
                float *dst_smem = cluster.map_shared_rank(smem_q, b);
                for (int j = threadIdx.x; j < data; j += num_threads) {
                    atomicAdd(&smem_q[j], dst_smem[j]);
                }
            }
        }
    }
    cluster.sync();
}

int main(int argc, char const **argv) {
    std::cout << "Parsed parameters:" << std::endl;
    std::cout << "  cluster_size: " << cluster_size << std::endl;
    std::cout << "  num_blocks: " << num_blocks << std::endl;
    std::cout << "  num_threads: " << num_threads << std::endl;
    std::cout << "  num_float_data: " << data << std::endl;
    std::cout << "  num_repeat: " << repeats << std::endl;

    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(num_blocks);
    config.blockDim = dim3(num_threads);


    float *global_output;

    cudaMalloc(&global_output, data * sizeof(float));

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x =  cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;
    config.dynamicSmemBytes = sizeof(SharedStorage<cluster_size>);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warmup = 50;
    int num_iterations = 100;
    auto func = clusterHist_kernel<1>;
    if (cluster_size == 1) {

    }
    else if (cluster_size == 2) {
        func = clusterHist_kernel<2>;
    }
    else if (cluster_size == 4) {
        func = clusterHist_kernel<4>;
    }
    else if (cluster_size == 8) {
        func = clusterHist_kernel<8>;
    }
    else if (cluster_size == 16) {
        func = clusterHist_kernel<16>;
    }

    cudaFuncSetAttribute((void*) func, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);
    cudaFuncSetAttribute((void*) func, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    for (int iter = 0; iter < warmup; ++iter) {
        cudaLaunchKernelEx(&config, func, global_output);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "clusterHist_kernel() execution failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaDeviceSynchronize();


    cudaEventRecord(start);

    for (int iter = 0; iter < num_iterations; ++iter) {
        cudaLaunchKernelEx(&config, func, global_output);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float average_time = milliseconds / num_iterations / 1000.0;

    long double total_data_bytes = num_blocks * data * repeats * sizeof(float);
    long double total_data_gb = total_data_bytes / (1024.0f * 1024.0f * 1024.0f);
    long double bandwidth_gbps = total_data_gb / average_time;

    std::cout << "Average time per kernel execution: " << average_time << " s\n";
    std::cout << "Total data size: " << total_data_gb << " GB\n";
    std::cout << "Bandwidth: " << bandwidth_gbps << " GB/s\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        std::cerr << "clusterHist_kernel() execution failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaDeviceSynchronize();

    cudaFree(global_output);

    return 0;
}