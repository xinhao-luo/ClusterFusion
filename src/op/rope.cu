#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>

#define BLOCK_SIZE 512
#define CLUSTER_SIZE 4

#define BATCH_SIZE 2
#define HEAD_DIM 2    // attn head dimension
#define HEAD_NUM 32     // attn head number
#define FFN_HIDDEN 512      // ffn hidden dimension
#define EMBEDDING_DIM 256   // token embedding dimension
#define SEQ_LEN 8        // seqence length

std::mt19937 rng(42);
std::normal_distribution<float> norm_dist(0.0, 5.0);

void fill_matrix(half* mat, int sz) {
    for (int i = 0; i < sz; i++) {
        float random_value = norm_dist(rng);
        if constexpr(std::is_same<half, __half>::value) {
            mat[i] = __float2half(random_value); // convert needed
        } else {
            mat[i] = random_value; 
        }
    }
}

inline std::vector<half> rope_cpu(
    const half* input,
    int dim,
    int encode_point_offset,
    float rope_scale, float rope_theta
) {
    int D = dim * HEAD_DIM;
    std::vector<half> rst(D * BATCH_SIZE);

    for (int i = 0; i < BATCH_SIZE; ++i) {
        std::vector<float> permuted_input(D);

        for (int k = 0; k < D; ++k) {
            permuted_input[k] = (k < D / 2) ? -__half2float(input[D * i + k + D / 2]) : __half2float(input[D * i + k - D / 2]);
        }

        for (int k = 0; k < D; ++k) {
            float inv_freq =
                (encode_point_offset / rope_scale) / (std::pow(rope_theta, float(2 * (k % (D / 2))) / float(D)));
            float cos = std::cos(inv_freq);
            float sin = std::sin(inv_freq);
            rst[D * i + k] = __float2half(cos * __half2float(input[D * i + k]) + sin * permuted_input[k]);
        }
    }

    return std::move(rst);
}

__global__ void __cluster_dims__(1, CLUSTER_SIZE, 1) 
rope(
    half* output,   // [BATCH_SIZE x dim x HEAD_DIM]
    half* input,  
    int dim,
    int encode_point_offset,
    float rope_scale, float rope_theta
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
    
}


int main(int argc, char** argv){
    half *h_input, *d_input;
    int encode_point_offset = 0;
    float rope_scale = 1;
    float rope_theta = 500000;
    h_input = new half[BATCH_SIZE * EMBEDDING_DIM * HEAD_DIM];

    fill_matrix(h_input, BATCH_SIZE * EMBEDDING_DIM * HEAD_DIM);

    // *--------------------------------------------------
    for(int i = 0;i < BATCH_SIZE * EMBEDDING_DIM * HEAD_DIM; ++i){
        printf("%f ", __half2float(h_input[i]));
    }
    printf("\n");
    // *--------------------------------------------------

    // ! CPU version
    // ! ===================================================
    std::vector<half> output = rope_cpu(h_input, EMBEDDING_DIM, encode_point_offset,rope_scale,rope_theta);
    for (int i = 0; i < output.size(); ++i) {
            std::cout << __half2float(output[i]) << " ";
    }
    printf("\n");
    // ! ===================================================


    return 0;
}