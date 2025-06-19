#include <iostream>
#include <random>
#include "dsm_microkernel.cuh"
namespace cg = cooperative_groups;

// nvcc --generate-code=arch=compute_90a,code=sm_90a -O3 -std=c++17 -lcuda microkernel.cu -o test && ./test

#define REDUCE_NUM 4096
#define CLUSTER_NUM 32
#define CLUSTER_SIZE 4

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 0.1);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(0.1f);
        }   
    }   
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) microkernel_global(
    half* output, // 1 * CLUSTER_NUM * REDUCE_NUM
    half* input  // 1 * CLUSTER_NUM * REDUCE_NUM
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t cluster_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();

    __shared__ __align__(16) half input_shmem[REDUCE_NUM];
    half __align__(16) reg_input[8];

    // 
    #pragma unroll
    for (int i = tid * 8; i < REDUCE_NUM; i+=BLOCK_SIZE * 8) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[i]);
        #pragma unroll
        for (int di = 0; di < 8; di++) 
            atomicAdd((&output[cluster_id * REDUCE_NUM + i + di]), reg_input[di]);
    }
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) microkernel_dsm(
    half* output, // 1 * CLUSTER_NUM * REDUCE_NUM
    half* input  // 1 * CLUSTER_NUM * REDUCE_NUM
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t cluster_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();

    // Init shared memory
    __shared__ __align__(16) half input_shmem[REDUCE_NUM];
    __shared__ alignas(128) half reduce_buffer[REDUCE_NUM];

    // Init registers
    uint32_t size, src_addr, dst_addr, neighbor_dst_bar = 0;

    // Init barrier
    __shared__ uint64_t barrier;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));

    // // Load input to shared memory
    // #pragma unroll
    // for (int i = tid * 8; i < REDUCE_NUM; i+=BLOCK_SIZE * 8) {
    //     *(uint4*)(&input_shmem[i]) = *(uint4*)(&input[cluster_block_id * CLUSTER_NUM * REDUCE_NUM + cluster_id * REDUCE_NUM + i]);
    // }
    // cluster.sync();

    // DSM Ring All-reduce
    size = REDUCE_NUM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(input_shmem));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(reduce_buffer));
    dsm_ring_allreduce<CLUSTER_SIZE, Stage::REDUCE>(
        size, tid, REDUCE_NUM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, input_shmem, reduce_buffer);
    
    if (cluster_block_id == 0) {
        #pragma unroll
        for (int i = tid * 8; i < REDUCE_NUM; i+=BLOCK_SIZE * 8) {
            *(uint4*)(&output[cluster_id * REDUCE_NUM + i]) = *(uint4*)(&input_shmem[i]);
        }
    }
        
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) microkernel_gather_global(
    half* output, 
    half* input 
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t cluster_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();

    __shared__ __align__(16) half input_shmem[REDUCE_NUM * CLUSTER_SIZE];

    // // Load input to shared memory
    // #pragma unroll
    // for (int i = tid * 8; i < REDUCE_NUM; i+=BLOCK_SIZE * 8) {
    //     *(uint4*)(&input_shmem[cluster_block_id * REDUCE_NUM + i]) = *(uint4*)(&input[cluster_id * CLUSTER_SIZE * REDUCE_NUM + cluster_block_id * REDUCE_NUM + i]);
    // }
    // cluster.sync();

    // 
    for (int i = tid * 8; i < REDUCE_NUM; i+=BLOCK_SIZE * 8) {
        *(uint4*)(&output[cluster_id * CLUSTER_SIZE * REDUCE_NUM + cluster_block_id * REDUCE_NUM + i]) = *(uint4*)(&input_shmem[cluster_block_id * REDUCE_NUM + i]);
    }
    cluster.sync();

    for (int i = tid * 8; i < REDUCE_NUM * CLUSTER_SIZE; i+=BLOCK_SIZE * 8) {
        *(uint4*)(&input_shmem[i]) = *(uint4*)(&output[cluster_id * CLUSTER_SIZE * REDUCE_NUM + i]);
    }
    cluster.sync();
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) microkernel_gather_dsm(
    half* output, // 1 * CLUSTER_NUM * REDUCE_NUM
    half* input  // 1 * CLUSTER_NUM * REDUCE_NUM
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t cluster_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();

    // Init shared memory
    __shared__ __align__(16) half input_shmem[REDUCE_NUM * CLUSTER_SIZE];
    __shared__ alignas(128) half reduce_buffer[REDUCE_NUM];

    // Init registers
    uint32_t size, src_addr, dst_addr, neighbor_dst_bar = 0;

    // Init barrier
    __shared__ uint64_t barrier;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));

    // // Load input to shared memory
    // #pragma unroll
    // for (int i = tid * 8; i < REDUCE_NUM; i+=BLOCK_SIZE * 8) {
    //     *(uint4*)(&input_shmem[i]) = *(uint4*)(&input[cluster_block_id * CLUSTER_NUM * REDUCE_NUM + cluster_id * REDUCE_NUM + i]);
    // }
    // cluster.sync();

    // DSM Ring All-reduce
    size = REDUCE_NUM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&input_shmem[cluster_block_id * REDUCE_NUM]));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&input_shmem[cluster_block_id * REDUCE_NUM]));
    dsm_ring_allreduce<CLUSTER_SIZE, Stage::GATHER>(
        size, tid, REDUCE_NUM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, input_shmem, input_shmem);
        
}

int main(int argc, char** argv) {   
    // Init input
    half *h_input, *d_input;
    half* h_output, *d_output;
    half* h_output_gather, *d_output_gather;
    h_input = new half[CLUSTER_SIZE * CLUSTER_NUM * REDUCE_NUM];
    h_output = new half[CLUSTER_NUM * REDUCE_NUM];
    h_output_gather = new half[CLUSTER_SIZE * CLUSTER_NUM * REDUCE_NUM];

    fill_matrix(h_input, CLUSTER_SIZE * CLUSTER_NUM * REDUCE_NUM);

    cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * CLUSTER_SIZE * CLUSTER_NUM * REDUCE_NUM);
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * CLUSTER_NUM * REDUCE_NUM);
    cudaMalloc(reinterpret_cast<void**>(&d_output_gather), sizeof(half) * CLUSTER_SIZE * CLUSTER_NUM * REDUCE_NUM);

    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * CLUSTER_SIZE * CLUSTER_NUM * REDUCE_NUM, cudaMemcpyHostToDevice);

    dim3 grid(CLUSTER_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    int wmup = 100;
    int test = 100;
    // for (int i = 0; i < wmup; i++) {
    //     microkernel_dsm<<<grid, block>>>(
    //         d_output,
    //         d_input
    //     );
    // }
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error: %s\n", cudaGetErrorString(err));
    // }
    // cudaDeviceSynchronize();

    // cudaEvent_t st, ed;
    // cudaEventCreate(&st);
    // cudaEventCreate(&ed);
    // cudaEventRecord(st);
    // for (int i = 0; i < test; i++) {
    //     microkernel_dsm<<<grid, block>>>(
    //         d_output,
    //         d_input
    //     );
    // }
    // cudaEventRecord(ed);
    // cudaEventSynchronize(ed);
    // float ms;
    // cudaEventElapsedTime(&ms, st, ed);
    // std::cout << "Latency: " << ms / test * 1e3 << " us" << std::endl;
    for (int i = 0; i < wmup; i++) {
        microkernel_gather_global<<<grid, block>>>(
            d_output_gather,
            d_input
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
        microkernel_gather_global<<<grid, block>>>(
            d_output_gather,
            d_input
        );
    }

    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / test * 1e3 << " us" << std::endl;
    // cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * CLUSTER_NUM * REDUCE_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_gather, reinterpret_cast<void*>(d_output_gather), sizeof(half) * CLUSTER_SIZE * CLUSTER_NUM * REDUCE_NUM, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < CLUSTER_SIZE * CLUSTER_NUM * REDUCE_NUM; i++)
    //     printf("%f, ", __half2float(h_output_gather[i]));
    // printf("\n");
    return 0;
}