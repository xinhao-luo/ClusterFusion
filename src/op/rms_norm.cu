#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>

#define BLOCK_SIZE 512
#define HEAD_DIM 1
#define HEAD_NUM 32
#define CLUSTER_SIZE 4
#define BATCH_SIZE 1
#define SEQ_LEN 128

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::mt19937 rng(42);
    std::normal_distribution<float> norm_dist(0.0, 5.0);
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
  std::vector<half> output(BATCH_SIZE * HEAD_DIM);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    float sum = 0;
    for (size_t j = 0; j < HEAD_DIM; ++j) {
      sum += float(input[i * HEAD_DIM + j]) * float(input[i * HEAD_DIM + j]);
    }
    float rms_rcp = 1.f / (std::sqrt(sum / float(HEAD_DIM)) + eps);
    for (size_t j = 0; j < HEAD_DIM; ++j) {
      output[i * HEAD_DIM + j] = (float(input[i * HEAD_DIM + j]) * rms_rcp) * float(weight[j]);
    }
  }
  return std::move(output);
}

__global__ void __cluster_dims__(1, CLUSTER_SIZE, 1) norm(
    half* output, 
    half* input,  
    half* w_rms
){
    namespace cg = cooperative_groups;
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t BATCH_SIZE_id         = blockIdx.x;
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % 32; // 32 per warp
    const uint32_t warp_id = tid / 32;

    // TODO

}

int main(int argc, char** argv){
    half *h_input, *d_input, *h_w, *d_w;

    h_input = new half[BATCH_SIZE * SEQ_LEN];
    h_w = new half[BATCH_SIZE * SEQ_LEN];

    fill_matrix(h_input, BATCH_SIZE * SEQ_LEN);
    fill_matrix(h_w, BATCH_SIZE * SEQ_LEN);

    // *--------------------------------------------------
    for(int i=0;i<BATCH_SIZE * SEQ_LEN;++i ){
        printf("%f ", __half2float(h_input[i]));
    }
    printf("\n");
    for(int i=0;i<BATCH_SIZE * SEQ_LEN;++i ){
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


    return 0;
}