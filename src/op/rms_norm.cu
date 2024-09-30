#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>


#define BLOCK_SIZE 512
#define CLUSTER_SIZE 4

#define BATCH_SIZE 1 
#define HEAD_DIM 128    // attn head dimension
#define HEAD_NUM 32     // attn head number
#define FFN_HIDDEN 512      // ffn hidden dimension
#define EMBEDDING_DIM 256   // token embedding dimension
#define SEQ_LEN 4096        // seqence length

std::mt19937 rng(42);
std::normal_distribution<float> norm_dist(0.0, 5.0);
template <typename T>
void fill_matrix(T* mat, int sz) {
    for (int i = 0; i < sz; i++) {
        float random_value = norm_dist(rng);
        if constexpr(std::is_same<T, __half>::value) {
            mat[i] = __float2half(random_value); // convert needed
        } else {
            mat[i] = random_value; 
        }
    }
}

inline std::vector<half> rms_norm(
    const half* input, const half* weight, float eps = 1e-5) {
  std::vector<half> output(BATCH_SIZE * EMBEDDING_DIM);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    float sum = 0;
    for (size_t j = 0; j < EMBEDDING_DIM; ++j) {
      sum += float(input[i * EMBEDDING_DIM + j]) * float(input[i * EMBEDDING_DIM + j]);
    }
    float rms_rcp = 1.f / (std::sqrt(sum / float(EMBEDDING_DIM)) + eps);
    for (size_t j = 0; j < EMBEDDING_DIM; ++j) {
      output[i * EMBEDDING_DIM + j] = __float2half((float(input[i * EMBEDDING_DIM + j]) * rms_rcp) * float(weight[j]));
    }
  }
  return std::move(output);
}

__global__ void __cluster_dims__(1, CLUSTER_SIZE, 1) 
norm(
    half* output, 
    half* input,  
    half* w_rms,
    float eps = 1e-5
){
    namespace cg = cooperative_groups;
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();

    const uint32_t batch_id         = blockIdx.x;
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % 32; // 32 per warp
    const uint32_t warp_id = tid / 32;

    // TODO: use DSM?
    // shared memory in block
    __shared__ float sum[BLOCK_SIZE/32]; 
    if (lane_id == 0){
        sum[warp_id] = 0;
    }

    __syncthreads();

    float local_sum = 0;
    // FIXME: only work with the fake loop, in need of extension
    half __align__(16) input_reg[8], weight_reg[8];
    // stage 1: read and calculate element-wise
    for (int d = tid; d < EMBEDDING_DIM / 8; d+=block.num_threads()) { // EMBEDDING_DIM <= 512 threads * 8
        *(uint4*)(&input_reg[0]) = *(uint4*)(&input[batch_id * EMBEDDING_DIM + d * 8]);
        *(uint4*)(&weight_reg[0]) = *(uint4*)(&w_rms[d * 8]);
    
        for (int di = 0; di < 8; di++) {
            local_sum += __half2float(input_reg[di])*__half2float(input_reg[di]);
        }
    }

    // stage 2: reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        // warp shuffle
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }

    if (lane_id == 0){
        sum[warp_id] = local_sum;
    }
    __syncthreads(); 

    if (tid==0){
        for (int i = 1; i < BLOCK_SIZE/32; i++){
            sum[0]+=sum[i];
        }
    }
    // finish block reduction here
    __syncthreads(); 

    // stage 3: update element-wise
    half rms_rcp = __float2half(1.f / (std::sqrt((sum[0]) / float(EMBEDDING_DIM)) + eps));

    for (int j = tid; j < EMBEDDING_DIM / 8; j+=block.num_threads()) {
        for (int di = 0; di < 8; di++) {
            input_reg[di]= __hmul(input_reg[di],(rms_rcp));
            output[batch_id * EMBEDDING_DIM + j * 8 + di] = __hmul(input_reg[di],weight_reg[di]);
        }
    }
    __syncthreads();
}


int main(int argc, char** argv){
    half *h_input, *d_input, *h_w, *d_w;

    h_input = new half[BATCH_SIZE * EMBEDDING_DIM];
    h_w = new half[EMBEDDING_DIM];

    fill_matrix(h_input, BATCH_SIZE * EMBEDDING_DIM);
    fill_matrix(h_w, EMBEDDING_DIM);

    // *--------------------------------------------------
    for(int i=0;i < BATCH_SIZE * EMBEDDING_DIM;++i ){
        printf("%f ", __half2float(h_input[i]));
    }
    printf("\n");
    for(int i=0;i < EMBEDDING_DIM;++i ){
        printf("%f ", __half2float(h_w[i]));
    }
    printf("\n");
    // *--------------------------------------------------

    // ! CPU version
    // ! ===================================================
    std::vector<half> output = rms_norm(h_input, h_w);
    for (size_t i = 0; i < output.size(); ++i) {
            std::cout << __half2float(output[i]) << " ";
    }
    // ! ===================================================

    half *d_output;
    cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * BATCH_SIZE * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w), sizeof(half) * EMBEDDING_DIM);
    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * BATCH_SIZE * EMBEDDING_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w), h_w, sizeof(half) * EMBEDDING_DIM, cudaMemcpyHostToDevice);

    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * BATCH_SIZE * EMBEDDING_DIM);
    
    /**
     * a cluster containing CLUSTER_SIZE blocks will be used to handle one head
     */
    dim3 grid(BATCH_SIZE, HEAD_NUM * CLUSTER_SIZE); // 32 * 4 blocks
    dim3 block(BLOCK_SIZE); 

    norm<<<grid, block, sizeof(float) * BLOCK_SIZE>>>(
        d_output,d_input,d_w
    );
    
    half* h_output = new __half[BATCH_SIZE * EMBEDDING_DIM]; 
    cudaMemcpy(h_output, d_output, sizeof(half) * BATCH_SIZE * EMBEDDING_DIM, cudaMemcpyDeviceToHost);

    for(int i=0;i< BATCH_SIZE * EMBEDDING_DIM;++i ){
        printf("%f ", __half2float(h_output[i]));
    }
    printf("\n");

    return 0;
}