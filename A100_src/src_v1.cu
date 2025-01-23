#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>
#include <math.h>

namespace cg = cooperative_groups;

// nvcc -arch=sm_80 -std=c++17 src.cu -o test && ./test

#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

/**
 * Kernel config
 */
#define BLOCK_SIZE 256
#define COOPERATE_BLOCK_NUM 32 // number of blocks working on a request (>HEAD_NUM)
/**
 * Decode config
 */
#define BATCH_SIZE 1
#define HEAD_DIM 128   // attn head dimension
#define HEAD_NUM 32      // attn head number

#define FFN_HIDDEN 4096     // ffn hidden dimension
#define EMBEDDING_DIM 4096  // token embedding dimension
#define SEQ_LEN 4096        // sequence length
#define ATTN_SPLIT 8*BLOCK_SIZE // 8 * 256, ATTN_SPLIT <= SEQ_LEN

std::mt19937 rng(42);
std::normal_distribution<float> norm_dist(0.0, 0.5);

template<typename T>
void fill_matrix(T *mat, int sz) {
    for (int i = 0; i < sz; i++) {
        float random_value;
        do {
            random_value = norm_dist(rng);
        } while (random_value < -0.1 || random_value > 0.1);
        if constexpr(std::is_same<T, __half>::value)
        {
            mat[i] = __float2half(random_value); // convert needed
        } else {
            mat[i] = random_value;
        }
    }
}
__device__ half dot(
        half *A,
        half *B,
        int len
) {
    half res = __float2half(0.0f);
#pragma unroll
    for (int i = 0; i < len; i++) {
        res += __hmul(A[i], B[i]);
    }
    return res;
}

/**
 * @param output
 * @param input
 * @param w_q
 * @param w_k
 * @param w_v
 * @param w_o
 * @param k_cache with rope
 * @param v_cache
 * @param ffn_1
 * @param ffn_2
 * @param ffn_3
 * @param w_rms1
 * @param w_rms2
 * @param rope_offset
 * @param rope_scale
 * @param sm_scale
 * @param rope_theta
 */
__global__ void decode(
        half *output, // batch * embedding_dim
        half *input,  // batch * 1 * embedding_dim
        half *w_q,    // head_num * (embedding_dim * head_dim)^T = head_num * head_dim * embedding_dim
        half *w_k,    // head_num * (embedding_dim * head_dim)^T = head_num * head_dim * embedding_dim
        half *w_v,    // head_num * (embedding_dim * head_dim)^T = head_num * head_dim * embedding_dim
        half *w_o,    // head_num * embedding_dim * head_dim
        half *k_cache,// batch * head_num * (seq_len - 1 (+1)) * head_dim
        half *v_cache,// batch * head_num * head_dim * (seq_len - 1 (+1))
        half *ffn_1,  // ffn_hidden * embedding_dim
        half *ffn_2,  // ffn_hidden * embedding_dim
        half *ffn_3,  // ffn_hidden * embedding_dim
        half *w_rms1, // embedding_dim
        half *w_rms2, // embedding_dim
        int rope_offset, // offset of RoPE starting point
        float rope_scale,
        float rope_theta,
        float sm_scale,
        half *global_reduction // batch * head_num * embedding_dim
) {
    float eps = 1e-5;

    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t head_idx = blockIdx.y;
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % 32; // 32 per warp
    const uint32_t warp_id = tid / 32;
    
    // preallocate SRAM. (be careful with SRAM usage)
    __shared__ float block_reduce_float[BLOCK_SIZE / 32];
    __shared__ __align__(16) half block_reduce_half[ATTN_SPLIT * 2]; // HEAD_DIM * 8 * 2 = 32 * 8 * 8
    __shared__ __align__(16) half shared_HEAD_DIM0[HEAD_DIM], shared_HEAD_DIM1[HEAD_DIM];
    
    __align__(16) half weight_reg[8], embedding_reg[16], residual_embedding_reg[16];
    __align__(16) half local_half1[8], local_half2[8];
    __align__(16) float local_float[8];

    // #------------------------------------------------------------------------
    // # RMS Norm
    // #------------------------------------------------------------------------
    *(uint4 * )(&residual_embedding_reg[0]) = *(uint4 * )(&input[batch_idx * EMBEDDING_DIM + tid * 16]);
    *(uint4 * )(&residual_embedding_reg[8]) = *(uint4 * )(&input[batch_idx * EMBEDDING_DIM + tid * 16 + 8]);
    float thread_sum = 0;
    #pragma unroll
    for (int di = 0; di < 16; di++) {
        thread_sum += __half2float(residual_embedding_reg[di]) * __half2float(residual_embedding_reg[di]);
    }
    // reduce in block
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, mask);
    }
    if (lane_id == 0) {
        block_reduce_float[warp_id] = thread_sum;
    }
    __syncthreads();
    if (tid == 0) {
        thread_sum = 0;
#pragma unroll
        for (int di = 0; di < BLOCK_SIZE / 32; di++) {
            thread_sum += block_reduce_float[di];
        }
        block_reduce_float[0] = __float2half(1.f / (std::sqrt(thread_sum / float(EMBEDDING_DIM))) + eps);
    }
    __syncthreads();

    half rms_rcp = block_reduce_float[0];
    *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&w_rms1[tid * 16]);
#pragma unroll
    for (int di = 0; di < 8; di++) {
        embedding_reg[di] = __hmul(__hmul(residual_embedding_reg[di], rms_rcp), weight_reg[di]);
    }
    *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&w_rms1[tid * 16 + 8]);
#pragma unroll
    for (int di = 0; di < 8; di++) {
        embedding_reg[di + 8] = __hmul(__hmul(residual_embedding_reg[di + 8], rms_rcp), weight_reg[di]);
    }
    // #------------------------------------------------------------------------

    // #------------------------------------------------------------------------
    // # W_q x -> rope
    // #------------------------------------------------------------------------
    for (int i = 0; i < HEAD_DIM; i++){
        thread_sum = 0;
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&w_q[head_idx * EMBEDDING_DIM * HEAD_DIM + i * EMBEDDING_DIM + tid * 16]);
        for (int j = 0; j < 8; ++j) {
            thread_sum += __half2float(embedding_reg[j]) * __half2float(weight_reg[j]);
        }
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&w_q[head_idx * EMBEDDING_DIM * HEAD_DIM + i * EMBEDDING_DIM + tid * 16 + 8]);
        for (int j = 0; j < 8; ++j) {
            thread_sum += __half2float(embedding_reg[j + 8]) * __half2float(weight_reg[j]);
        }
        // warp level reduce
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, mask);
        }
        if (lane_id == 0) {
            block_reduce_half[i * (BLOCK_SIZE / 32) + warp_id] = thread_sum;
        }
    }
    __syncthreads();
    // rope
    if(tid < HEAD_DIM / 2){
        *(uint4 * )(&local_half1[0]) = *(uint4 * )(&block_reduce_half[tid * (BLOCK_SIZE / 32)]);
        *(uint4 * )(&local_half2[0]) = *(uint4 * )(&block_reduce_half[(tid + HEAD_DIM / 2) * (BLOCK_SIZE / 32)]);
        half local1 = 0, local2 = 0;
        #pragma unroll
            for (int di = 0; di < BLOCK_SIZE / 32; di++) {
                local1 += local_half1[di];
                local2 += local_half2[di];
            }
        float inv_freq = (rope_offset / rope_scale) / (std::pow(rope_theta, float(2 * tid) / float(HEAD_DIM)));
        float cos = std::cos(inv_freq);
        float sin = std::sin(inv_freq);
        shared_HEAD_DIM0[tid] = __float2half(cos * __half2float(local1) - sin * __half2float(local2));
        shared_HEAD_DIM0[tid + HEAD_DIM / 2] = __float2half(cos * __half2float(local2) + sin * __half2float(local1));
    }
    __syncthreads();
    // #------------------------------------------------------------------------
    // * 0.0959488 ms

    // #------------------------------------------------------------------------
    // # W_k x -> rope
    // #------------------------------------------------------------------------
    for (int i = 0; i < HEAD_DIM; i++){
        thread_sum = 0;
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&w_k[head_idx * EMBEDDING_DIM * HEAD_DIM + i * EMBEDDING_DIM + tid * 16]);
        for (int j = 0; j < 8; ++j) {
            thread_sum += __half2float(embedding_reg[j]) * __half2float(weight_reg[j]);
        }
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&w_k[head_idx * EMBEDDING_DIM * HEAD_DIM + i * EMBEDDING_DIM + tid * 16 + 8]);
        for (int j = 0; j < 8; ++j) {
            thread_sum += __half2float(embedding_reg[j + 8]) * __half2float(weight_reg[j]);
        }
        // warp level reduce
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, mask);
        }
        if (lane_id == 0) {
            block_reduce_half[i * (BLOCK_SIZE / 32) + warp_id] = thread_sum;
        }
    }
    __syncthreads();
    // rope
    if(tid < HEAD_DIM / 2){
        *(uint4 * )(&local_half1[0]) = *(uint4 * )(&block_reduce_half[tid * (BLOCK_SIZE / 32)]);
        *(uint4 * )(&local_half2[0]) = *(uint4 * )(&block_reduce_half[(tid + HEAD_DIM / 2) * (BLOCK_SIZE / 32)]);
        half local1 = 0, local2 = 0;
        #pragma unroll
            for (int di = 0; di < BLOCK_SIZE / 32; di++) {
                local1 += local_half1[di];
                local2 += local_half2[di];
            }
        float inv_freq = (rope_offset / rope_scale) / (std::pow(rope_theta, float(2 * tid) / float(HEAD_DIM)));
        float cos = std::cos(inv_freq);
        float sin = std::sin(inv_freq);
        shared_HEAD_DIM1[tid] = __float2half(cos * __half2float(local1) - sin * __half2float(local2));
        shared_HEAD_DIM1[tid + HEAD_DIM / 2] = __float2half(cos * __half2float(local2) + sin * __half2float(local1));
    }
    __syncthreads();
    // #------------------------------------------------------------------------
    // * 0.183706 ms

    // #------------------------------------------------------------------------
    // # Attention
    // # ~~~~~~~~~~~~~~~~~
    // # Q: shared_HEAD_DIM0
    // # K: shared_HEAD_DIM1
    // #------------------------------------------------------------------------
    // load q from shared memory to local
    __align__(16) half local_head_dim[HEAD_DIM];
    for (int i = 0; i< HEAD_DIM / 8; ++i) {
        *(uint4 * )(&local_head_dim[8 * i]) = *(uint4 * )(&shared_HEAD_DIM0[8 * i]);
    }
    if (lane_id == 0) {
        block_reduce_float[warp_id] = 0;
    }
    float prev_l = 0, current_l;
    float prev_m = 0, current_m;
    // prev o: embedding_reg
    __align__(16) half local_attn_score[ATTN_SPLIT / 32];
    __align__(16) half local_attn[16];
    for (int iter = tid; iter < SEQ_LEN / 8; iter += BLOCK_SIZE) {
        // attention score
        float max_score = 0;
        for (int dj = 0; dj < 8; ++dj) {
            int line_id = iter * 8 + dj;
            float score = 0;
            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8; ++i) {
                if (line_id == SEQ_LEN - 1){
                    *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&shared_HEAD_DIM1[8 * i]);
                } else {
                    *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&k_cache[
                        batch_idx * HEAD_NUM * SEQ_LEN * HEAD_DIM + head_idx * SEQ_LEN * HEAD_DIM + line_id * HEAD_DIM + i * 8
                    ]);
                }
                #pragma unroll
                for (int di = 0; di < 8; di++) {
                    score += __half2float(local_head_dim[i * 8 + di]) * __half2float(weight_reg[di]);
                }
            }
            local_attn[dj] = __float2half(score/sm_scale);
            max_score = max(max_score,score);
        }
        *(uint4 * )(&block_reduce_half[tid * 8]) = *(uint4 * )(&local_attn[0]);
        // rowmax warp level reduce
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            max_score = max(max_score, __shfl_down_sync(0xffffffff, max_score, mask));
        }
        if (lane_id == 0) {
            block_reduce_float[warp_id] = max_score;
        }
        __syncthreads();
        // rowmax
        *(uint4 * )(&local_float[0]) = *(uint4 * )(&block_reduce_float[0]);
        current_m = prev_m;
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE / 32; ++i){ // 8
            current_m = max(current_m, local_float[i]);
        }
        // load score
        current_l = 0;
        #pragma unroll
        for (int i = 0; i < ATTN_SPLIT / 256; ++i){
            *(uint4 * )(&local_attn_score[8 * i]) = *(uint4 * )(&block_reduce_half[lane_id * ATTN_SPLIT / 32 + i * 8]);
            #pragma unroll
            for (int ki = 0; ki < 8; ki++) {
                float exp_ = exp(__half2float(local_attn_score[8 * i + ki]) - current_m); // safe softmax
                local_attn_score[8 * i + ki] = exp_;
                current_l += exp_;
            }
        }
        // warp reduce to lane_id 0 
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            current_l += __shfl_down_sync(0xffffffff, current_l, mask);
        }
        // broad cast
        current_l = __shfl_sync(0xffffffff, current_l, 0);
        current_l += prev_l * exp(prev_m - current_m);
        prev_m = current_m;
        if (head_idx == 0) {
            output[tid] = current_l;
        }
        // score * V
        // 16 x 64
        for (int i = 0; i < 16; ++i) {
            float attntion = 0;
            // starting column
            int column_id = (iter / (ATTN_SPLIT / 8)) * ATTN_SPLIT + lane_id * ATTN_SPLIT / 32;
            int row_id = warp_id * 16 + i; 
            for (int di = 0; di < ATTN_SPLIT / (32 * 8); ++di) { // 8
                *(uint4 * )(&local_half1[0]) = *(uint4 * )(&v_cache[
                    batch_idx * HEAD_NUM * SEQ_LEN * HEAD_DIM + 
                    head_idx * SEQ_LEN * HEAD_DIM + 
                    SEQ_LEN * row_id + column_id + 8 * di
                ]);
                if (column_id + 8 * di + 8 == SEQ_LEN) {
                    local_half1[7] = shared_HEAD_DIM1[16 * warp_id + i];
                }
                #pragma unroll
                for (int ki = 0; ki < 8; ki++) {
                    attntion += __half2float(local_attn_score[di * 8 + ki]) * __half2float(local_half1[ki]);
                }
            }
            local_attn[i] = attntion;
        }
        for (int i = 0; i < 16; ++i) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                local_attn[i] += __shfl_down_sync(0xffffffff, local_attn[i], mask);
            }
            local_attn[i] = __shfl_sync(0xffffffff, local_attn[i], 0);
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            embedding_reg[i] = __float2half(__half2float(embedding_reg[i]) * prev_l / current_l + __half2float(local_attn[i]) / current_l);
        }
        prev_l = current_l;
    }
    // #------------------------------------------------------------------------
    // * 0.462643
    // now, for each warp, embedding_reg contains a tile of attn value

    // #------------------------------------------------------------------------
    // # Attention Projection
    // #------------------------------------------------------------------------
    for (int iter = 0; iter < EMBEDDING_DIM / (32 * 8); ++iter) {
        for (int iter_inner = 0; iter_inner < 8; ++iter_inner) {
            *(uint4 * )(&local_half1[0]) = *(uint4 * )(&w_o[head_idx * EMBEDDING_DIM * HEAD_DIM + (32 * (8 * iter + iter_inner) + lane_id) * HEAD_DIM + warp_id * 16]);
            float score = 0;
            for (int i = 0; i < 8; ++i){
                score += __half2float(embedding_reg[i]) * __half2float(local_half1[i]);
            }
            *(uint4 * )(&local_half2[0]) = *(uint4 * )(&w_o[head_idx * EMBEDDING_DIM * HEAD_DIM + (32 * (8 * iter + iter_inner) + lane_id) * HEAD_DIM + warp_id * 16 + 8]);
            for (int i = 0; i < 8; ++i){
                score += __half2float(embedding_reg[i + 8]) * __half2float(local_half2[i]);
            }
            block_reduce_half[(iter_inner * 8 + lane_id) * 8 + warp_id] = __float2half(score);
        }
        __syncthreads();
        if (warp_id == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                *(uint4 * )(&local_half1[0]) = *(uint4 * )(&block_reduce_half[(lane_id * 8 + i) * 8]);
                half sum = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sum += local_half1[j];
                }
                local_half2[i] = sum;
            }
            *(uint4 * )(&global_reduction[batch_idx * HEAD_NUM * EMBEDDING_DIM + head_idx * EMBEDDING_DIM + (iter * 32 + lane_id) * 8]) = *(uint4 * )(&local_half2[0]);
        }
    }
    grid.sync();
    // #------------------------------------------------------------------------
    // * 0.570778
    // now, attention value of each head is written in global

    // #------------------------------------------------------------------------
    // # Residual and Norm
    // #------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < HEAD_NUM; ++i) {
        *(uint4 * )(&local_half1[0]) = *(uint4 * )(&global_reduction[batch_idx * HEAD_NUM * EMBEDDING_DIM + i * EMBEDDING_DIM + tid * 16]);
        #pragma unroll
        for (int di = 0; di < 8; ++di) {
            residual_embedding_reg[di] += local_half1[di];
        } 
        *(uint4 * )(&local_half2[0]) = *(uint4 * )(&global_reduction[batch_idx * HEAD_NUM * EMBEDDING_DIM + i * EMBEDDING_DIM + tid * 16 + 8]);
        #pragma unroll
        for (int di = 0; di < 8; ++di) {
            residual_embedding_reg[di + 8] += local_half2[di];
        } 
    }
    thread_sum = 0;
    #pragma unroll
    for (int di = 0; di < 16; di++) {
        thread_sum += __half2float(residual_embedding_reg[di]) * __half2float(residual_embedding_reg[di]);
    }
    // reduce in block
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, mask);
    }
    if (lane_id == 0) {
        block_reduce_float[warp_id] = thread_sum;
    }
    __syncthreads();
    if (tid == 0) {
        thread_sum = 0;
    #pragma unroll
        for (int di = 0; di < BLOCK_SIZE / 32; di++) {
            thread_sum += block_reduce_float[di];
        }
        block_reduce_float[0] = __float2half(1.f / (std::sqrt(thread_sum / float(EMBEDDING_DIM))) + eps);
    }
    __syncthreads();
    
    rms_rcp = block_reduce_float[0];
    *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&w_rms2[tid * 16]);
    #pragma unroll
    for (int di = 0; di < 8; di++) {
        embedding_reg[di] = __hmul(__hmul(residual_embedding_reg[di], rms_rcp), weight_reg[di]);
    }
    *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&w_rms1[tid * 16 + 8]);
    #pragma unroll
    for (int di = 0; di < 8; di++) {
        embedding_reg[di + 8] = __hmul(__hmul(residual_embedding_reg[di + 8], rms_rcp), weight_reg[di]);
    }
    // #------------------------------------------------------------------------
    // * 0.583782

    // #------------------------------------------------------------------------
    // # FFN
    // #------------------------------------------------------------------------
    // ffn 1 + silu * ffn 3
    for (int i = 0; i < FFN_HIDDEN / COOPERATE_BLOCK_NUM; i++){
        float thread_sum_ffn1 = 0, thread_sum_ffn3 = 0;
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&ffn_1[(head_idx * (FFN_HIDDEN / COOPERATE_BLOCK_NUM) + i) * EMBEDDING_DIM + tid * 16]);
        for (int j = 0; j < 8; ++j) {
            thread_sum_ffn1 += __half2float(embedding_reg[j]) * __half2float(weight_reg[j]);
        }
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&ffn_1[(head_idx * (FFN_HIDDEN / COOPERATE_BLOCK_NUM) + i) * EMBEDDING_DIM + tid * 16 + 8]);
        for (int j = 0; j < 8; ++j) {
            thread_sum_ffn1 += __half2float(embedding_reg[j + 8]) * __half2float(weight_reg[j]);
        }
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&ffn_3[(head_idx * (FFN_HIDDEN / COOPERATE_BLOCK_NUM) + i) * EMBEDDING_DIM + tid * 16]);
        for (int j = 0; j < 8; ++j) {
            thread_sum_ffn3 += __half2float(embedding_reg[j]) * __half2float(weight_reg[j]);
        }
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&ffn_3[(head_idx * (FFN_HIDDEN / COOPERATE_BLOCK_NUM) + i) * EMBEDDING_DIM + tid * 16 + 8]);
        for (int j = 0; j < 8; ++j) {
            thread_sum_ffn3 += __half2float(embedding_reg[j + 8]) * __half2float(weight_reg[j]);
        }
        // warp level reduce
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            thread_sum_ffn1 += __shfl_down_sync(0xffffffff, thread_sum_ffn1, mask);
            thread_sum_ffn3 += __shfl_down_sync(0xffffffff, thread_sum_ffn3, mask);
        }
        if (lane_id == 0) {
            block_reduce_half[i * (BLOCK_SIZE / 32) + warp_id] = thread_sum_ffn1;
            block_reduce_half[FFN_HIDDEN / COOPERATE_BLOCK_NUM * BLOCK_SIZE / 32 + i * (BLOCK_SIZE / 32) + warp_id] = thread_sum_ffn3;
        }
    }
    __syncthreads();

    if(tid < FFN_HIDDEN / COOPERATE_BLOCK_NUM){
        *(uint4 * )(&local_half1[0]) = *(uint4 * )(&block_reduce_half[tid * (BLOCK_SIZE / 32)]);
        *(uint4 * )(&local_half2[0]) = *(uint4 * )(&block_reduce_half[(FFN_HIDDEN / COOPERATE_BLOCK_NUM + tid) * BLOCK_SIZE / 32]);
        half local1 = 0, local2 = 0;
        #pragma unroll
        for (int di = 0; di < BLOCK_SIZE / 32; di++) {
            local1 += local_half1[di];
            local2 += local_half2[di];
        }
        float tmp = __half2float(local1);
        tmp /= (1.0f + expf(-tmp));
        shared_HEAD_DIM0[tid] = __float2half(tmp * __half2float(local2));
    }
    __syncthreads();
    // * 0.75223

    // ffn 2
    __align__(16) half local_ffn_tile[FFN_HIDDEN / COOPERATE_BLOCK_NUM];
    for (int i = 0; i < (FFN_HIDDEN / COOPERATE_BLOCK_NUM) / 8; ++i) {
        *(uint4 * )(&local_ffn_tile[8 * i]) = *(uint4 * )(&shared_HEAD_DIM0[8 * i]);
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        embedding_reg[i] = __float2half(0.0f);
    }
    for (int i = 0; i < (FFN_HIDDEN / COOPERATE_BLOCK_NUM); ++i) {
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&ffn_2[
            (head_idx * (FFN_HIDDEN / COOPERATE_BLOCK_NUM) + i) * EMBEDDING_DIM + 16 * tid 
        ]);
        #pragma unroll
        for (int di = 0; di < 8; ++di) {
            embedding_reg[di] += __half2float(local_ffn_tile[i]) * __half2float(weight_reg[di]);
        }
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&ffn_2[
            (head_idx * (FFN_HIDDEN / COOPERATE_BLOCK_NUM) + i) * EMBEDDING_DIM + 16 * tid + 8
        ]);
        #pragma unroll
        for (int di = 0; di < 8; ++di) {
            embedding_reg[di + 8] += __half2float(local_ffn_tile[i]) * __half2float(weight_reg[di]);
        }
    }
    // * 0.756122
    *(uint4 * )(&global_reduction[batch_idx * HEAD_NUM * EMBEDDING_DIM + head_idx * EMBEDDING_DIM + tid * 16]) = *(uint4 * )(&embedding_reg[0]);
    *(uint4 * )(&global_reduction[batch_idx * HEAD_NUM * EMBEDDING_DIM + head_idx * EMBEDDING_DIM + tid * 16 + 8]) = *(uint4 * )(&embedding_reg[8]);
    grid.sync();
    // #------------------------------------------------------------------------
    // * 0.812032

    // #------------------------------------------------------------------------
    // # Reduce and Residual
    // #------------------------------------------------------------------------
    if (head_idx == 0) {
        for (int i = 0; i < COOPERATE_BLOCK_NUM; ++i) {
            *(uint4 * )(&local_half1[0]) = *(uint4 * )(&global_reduction[batch_idx * HEAD_NUM * EMBEDDING_DIM + i * EMBEDDING_DIM + tid * 16]);
            #pragma unroll
            for (int di = 0; di < 8; ++di) {
                residual_embedding_reg[di] += local_half1[di];
            } 
            *(uint4 * )(&local_half2[0]) = *(uint4 * )(&global_reduction[batch_idx * HEAD_NUM * EMBEDDING_DIM + i * EMBEDDING_DIM + tid * 16 + 8]);
            #pragma unroll
            for (int di = 0; di < 8; ++di) {
                residual_embedding_reg[di + 8] += local_half2[di];
            } 
        }
        *(uint4 * )(&output[batch_idx * EMBEDDING_DIM + tid * 16]) = *(uint4 * )(&residual_embedding_reg[0]);
        *(uint4 * )(&output[batch_idx * EMBEDDING_DIM + tid * 16 + 8]) = *(uint4 * )(&residual_embedding_reg[8]);
    }
    // * 0.824115
}

int main(int argc, char **argv) {
    // shared memory size per threadBlock
    size_t dynamicShMemSize = sizeof(float) * (BLOCK_SIZE / 16) + sizeof(half) * (ATTN_SPLIT * 2 + HEAD_DIM * 2);
    cudaFuncSetAttribute(decode, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicShMemSize);

    int rope_offset = SEQ_LEN;
    float rope_scale = 1;
    float rope_theta = 1e4;
    float sm_scale = sqrt(HEAD_DIM);
    half *h_input, *d_input;
    half *h_k_cache, *d_k_cache;
    half *h_v_cache, *d_v_cache;
    half *h_w_q, *d_w_q;
    half *h_w_k, *d_w_k;
    half *h_w_v, *d_w_v;
    half *h_w_o, *d_w_o;
    half *h_ffn_1, *d_ffn_1;
    half *h_ffn_2, *d_ffn_2;
    half *h_ffn_3, *d_ffn_3;
    half *h_rms_1, *d_rms_1;
    half *h_rms_2, *d_rms_2;
    h_input = new half[BATCH_SIZE * 1 * EMBEDDING_DIM];
    h_w_q = new half[HEAD_NUM * EMBEDDING_DIM * HEAD_DIM];
    h_w_k = new half[HEAD_NUM * EMBEDDING_DIM * HEAD_DIM];
    h_w_v = new half[HEAD_NUM * EMBEDDING_DIM * HEAD_DIM];
    h_w_o = new half[HEAD_NUM * HEAD_DIM * EMBEDDING_DIM];
    h_k_cache = new half[BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM];
    h_v_cache = new half[BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM];
    h_ffn_1 = new half[EMBEDDING_DIM * FFN_HIDDEN];
    h_ffn_2 = new half[FFN_HIDDEN * EMBEDDING_DIM];
    h_ffn_3 = new half[EMBEDDING_DIM * FFN_HIDDEN];
    h_rms_1 = new half[EMBEDDING_DIM];
    h_rms_2 = new half[EMBEDDING_DIM];

    fill_matrix(h_input, BATCH_SIZE * 1 * EMBEDDING_DIM);
    fill_matrix(h_w_q, HEAD_NUM * EMBEDDING_DIM * HEAD_DIM);
    fill_matrix(h_w_k, HEAD_NUM * EMBEDDING_DIM * HEAD_DIM);
    fill_matrix(h_w_v, HEAD_NUM * EMBEDDING_DIM * HEAD_DIM);
    fill_matrix(h_w_o, HEAD_NUM * HEAD_DIM * EMBEDDING_DIM);
    fill_matrix(h_k_cache, BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM);
    fill_matrix(h_v_cache, BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM);
    fill_matrix(h_ffn_1, EMBEDDING_DIM * FFN_HIDDEN);
    fill_matrix(h_ffn_2, FFN_HIDDEN * EMBEDDING_DIM);
    fill_matrix(h_ffn_3, EMBEDDING_DIM * FFN_HIDDEN);
    fill_matrix(h_rms_1, EMBEDDING_DIM);
    fill_matrix(h_rms_2, EMBEDDING_DIM);

    cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(half) * BATCH_SIZE * 1 * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_w_q), sizeof(half) * HEAD_NUM * EMBEDDING_DIM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_w_k), sizeof(half) * HEAD_NUM * EMBEDDING_DIM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_w_v), sizeof(half) * HEAD_NUM * EMBEDDING_DIM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_w_o), sizeof(half) * HEAD_NUM * HEAD_DIM * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_k_cache), sizeof(half) * BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_v_cache), sizeof(half) * BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_1), sizeof(half) * EMBEDDING_DIM * FFN_HIDDEN);
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_2), sizeof(half) * FFN_HIDDEN * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_3), sizeof(half) * EMBEDDING_DIM * FFN_HIDDEN);
    cudaMalloc(reinterpret_cast<void **>(&d_rms_1), sizeof(half) * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_rms_2), sizeof(half) * EMBEDDING_DIM);

    cudaMemcpy(reinterpret_cast<void *>(d_input), h_input, sizeof(half) * BATCH_SIZE * 1 * EMBEDDING_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_q), h_w_q, sizeof(half) * HEAD_NUM * EMBEDDING_DIM * HEAD_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_k), h_w_k, sizeof(half) * HEAD_NUM * EMBEDDING_DIM * HEAD_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_v), h_w_v, sizeof(half) * HEAD_NUM * EMBEDDING_DIM * HEAD_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_o), h_w_o, sizeof(half) * HEAD_NUM * HEAD_DIM * EMBEDDING_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_k_cache), h_k_cache,
               sizeof(half) * BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_v_cache), h_v_cache,
               sizeof(half) * BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_ffn_1), h_ffn_1, sizeof(half) * EMBEDDING_DIM * FFN_HIDDEN,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_ffn_2), h_ffn_2, sizeof(half) * FFN_HIDDEN * EMBEDDING_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_ffn_3), h_ffn_3, sizeof(half) * EMBEDDING_DIM * FFN_HIDDEN,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_rms_1), h_rms_1, sizeof(half) * EMBEDDING_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_rms_2), h_rms_2, sizeof(half) * EMBEDDING_DIM, cudaMemcpyHostToDevice);

    half *h_output, *d_output;
    h_output = new half[BATCH_SIZE * EMBEDDING_DIM];
    cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(half) * BATCH_SIZE * EMBEDDING_DIM);

    half *global_reduce;
    cudaMalloc(reinterpret_cast<void **>(&global_reduce), sizeof(half) * BATCH_SIZE * HEAD_NUM * EMBEDDING_DIM);
    
    void *kernelArgs[] = {
            &d_output, &d_input,
            &d_w_q, &d_w_k, &d_w_v, &d_w_o,
            &d_k_cache, &d_v_cache,
            &d_ffn_1, &d_ffn_2, &d_ffn_3,
            &d_rms_1, &d_rms_2,
            &rope_offset, &rope_scale, &rope_theta,
            &sm_scale,
            &global_reduce
    };

    /**
     * ! Without cluster, we have to do heavy global sync in grid scope.
     *      So we try to do less intro-block sync in the kernel.
     */
    dim3 grid(BATCH_SIZE, HEAD_NUM, COOPERATE_BLOCK_NUM/HEAD_NUM);
    dim3 block(BLOCK_SIZE);
    int warmup = 50;
    int test = 10;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);

    // * warm up
    for (int i = 0; i < warmup; i++) {
        cudaLaunchCooperativeKernel((void *) decode, grid, block, kernelArgs, dynamicShMemSize);
    }

    // * test kernel
    cudaEventRecord(st);
    for (int i = 0; i < test; i++) {
        cudaLaunchCooperativeKernel((void *) decode, grid, block, kernelArgs, dynamicShMemSize);
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);

    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << (ms / (1.0 * test)) << " ms" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void *>(d_output), sizeof(half) * BATCH_SIZE * EMBEDDING_DIM,
               cudaMemcpyDeviceToHost);

    return 0;
}