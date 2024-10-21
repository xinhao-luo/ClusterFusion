#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>
namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define COOPERATE_BLOCK_NUM 32 // number of blocks working on a request (>HEAD_NUM)

#define BATCH_SIZE 2
#define HEAD_DIM 32        // attn head dimension
#define HEAD_NUM 2         // attn head number
#define FFN_HIDDEN 4096     // ffn hidden dimension
#define EMBEDDING_DIM 16  // token embedding dimension
#define SEQ_LEN 4096        // sequence length
#define FL_DEC_SPLIT 256

std::mt19937 rng(42);
std::normal_distribution<float> norm_dist(0.0, 1.0);

template<typename T>
void fill_matrix(T *mat, int sz) {
    for (int i = 0; i < sz; i++) {
        float random_value = norm_dist(rng);
        if constexpr(std::is_same<T, __half>::value)
        {
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
    int D = dim;
    std::vector<half> rst(HEAD_NUM * BATCH_SIZE * D);

    for (int i = 0; i < HEAD_NUM * BATCH_SIZE; ++i) {
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

__global__ void rope(
    half *output, half *input,
    int pos_offset,
    float rope_scale, float rope_theta
) {
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t head_idx = blockIdx.y;
    const uint32_t cooperate_idx = blockIdx.z;
    const uint32_t tid = cg::this_thread_block().thread_rank();
    int d = BLOCK_SIZE * cooperate_idx + tid;

    __align__(16)
    half input_reg[8], permuted_input_reg[8], roped_output_reg[8];
    if (d < HEAD_DIM / 8) {
        *(uint4 * )(&input_reg[0]) = *(uint4 * )(&input[batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM + d * 8]);
        int permuted_idx = d * 8 >= HEAD_DIM / 2 ? (d * 8 - HEAD_DIM / 2) : (d * 8 + HEAD_DIM / 2);
        *(uint4 * )(&permuted_input_reg[0]) = *(uint4 * )(
                &input[batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM + permuted_idx]
        );
        for (int k = 0; k < 8; ++k) {
            half permuted = d * 8 >= HEAD_DIM / 2 ? permuted_input_reg[k] : -permuted_input_reg[k];
            int idx = d * 8 + k;
            float inv_freq =
                    (pos_offset / rope_scale) /
                    (std::pow(rope_theta, float(2 * (idx % (HEAD_DIM / 2))) / float(HEAD_DIM)));
            float cos = std::cos(inv_freq);
            float sin = std::sin(inv_freq);
            roped_output_reg[k] = __float2half(
                    cos * __half2float(input_reg[k]) + sin * __half2float(permuted)
            );
        }
        *(uint4 * )(&output[batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM + d * 8])
                = *(uint4 * )(roped_output_reg);
    }
}


int main(int argc, char** argv){
    half *h_input, *d_input;
    int encode_point_offset = SEQ_LEN;
    float rope_scale = 1;
    float rope_theta = 500000;
    h_input = new half[BATCH_SIZE * HEAD_NUM * HEAD_DIM];

    fill_matrix(h_input, BATCH_SIZE * HEAD_NUM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(half) * BATCH_SIZE * HEAD_NUM * HEAD_DIM);
    cudaMemcpy(reinterpret_cast<void *>(d_input), h_input, sizeof(half) * BATCH_SIZE * HEAD_NUM * HEAD_DIM,
    cudaMemcpyHostToDevice);
    // *--------------------------------------------------
    for(int i = 0;i < BATCH_SIZE * HEAD_NUM * HEAD_DIM; ++i){
        printf("%f ", __half2float(h_input[i]));
    }
    printf("\n");
    // *--------------------------------------------------

    // ! CPU version
    // ! ===================================================
    std::vector<half> output = rope_cpu(h_input, HEAD_DIM, encode_point_offset,rope_scale,rope_theta);
    for (int i = 0; i < output.size(); ++i) {
            std::cout << __half2float(output[i]) << " ";
    }
    printf("\n");
    // ! ===================================================

    half h_output[ BATCH_SIZE * HEAD_NUM * HEAD_DIM] = {__float2half(0.0)};
    half *d_output;
    cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(half) * BATCH_SIZE * HEAD_NUM * HEAD_DIM);
    cudaMemcpy(reinterpret_cast<void *>(d_output), h_output,
    sizeof(half) * BATCH_SIZE * HEAD_NUM * HEAD_DIM, cudaMemcpyHostToDevice);
    void *kernelArgs[] = {
        &d_output, &d_input, 
        &encode_point_offset, &rope_scale, &rope_theta
    };
    dim3 grid(BATCH_SIZE, HEAD_NUM, COOPERATE_BLOCK_NUM / HEAD_NUM);
    dim3 block(BLOCK_SIZE);
    cudaLaunchCooperativeKernel((void *) rope, grid, block, kernelArgs);

    cudaMemcpy(h_output, reinterpret_cast<void *>(d_output), sizeof(half) * BATCH_SIZE * HEAD_NUM * HEAD_DIM,
               cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < BATCH_SIZE * HEAD_NUM * HEAD_DIM; ++i) {
        std::cout << __half2float(h_output[i]) << " ";
    }
    return 0;
}