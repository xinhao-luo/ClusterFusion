#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>
#include <math.h>

namespace cg = cooperative_groups;

// nvcc -arch=sm_80 -std=c++17 src.cu -o test -Xptxas=-v -Xptxas=-warn-lmem-usage -maxrregcount=128

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

#define MAX(a, b) ((a) > (b) ? (a) : (b))

/**
 * Kernel config
 */
#define ATTN_BLOCK_SIZE 512
#define COOPERATE_BLOCK_NUM 32 // number of blocks working on a request (>HEAD_NUM)

#define FFN_BLOCK_SIZE 1024 // >= 512

#define WARP_SIZE 32

/**
 * Constant
 */
#define ROPE_OFFSET SEQ_LEN
#define ROPE_SCALE float(1)
#define ROPE_THETA float(1e4)

/**
 * Decode config
 */
#define BATCH_SIZE 1
#define HEAD_DIM 128   // attn head dimension
#define HEAD_NUM 32      // attn head number

#define FFN_HIDDEN 4096     // ffn hidden dimension
#define EMBEDDING_DIM 4096  // token embedding dimension
#define SEQ_LEN 4096        // sequence length
#define ATTN_SPLIT (8*ATTN_BLOCK_SIZE) // ATTN_SPLIT <= SEQ_LEN

#define BLOCK_REDUCE_HALF MAX(MAX(2 * HEAD_DIM * ATTN_BLOCK_SIZE / WARP_SIZE, ATTN_SPLIT), FFN_HIDDEN / COOPERATE_BLOCK_NUM * ATTN_BLOCK_SIZE / WARP_SIZE * 2)

std::mt19937 rng(42);
std::normal_distribution<float> norm_dist(0.0, 0.1);

template<typename T>
void fill_matrix(T *mat, int sz) {
    for (int i = 0; i < sz; i++) {
        float random_value;
        do {
            random_value = norm_dist(rng);
        } while (random_value < -0.1 || random_value > 0.1 || (random_value > -0.05 && random_value < 0.05));
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
 * @param sm_scale
 */
 __global__ void decode_part1(
    half *output, // batch * embedding_dim
    half *input,  // batch * 1 * embedding_dim
    half *w_q,    // head_num * (embedding_dim * head_dim)^T = head_num * head_dim * embedding_dim
    half *w_k,    // head_num * (embedding_dim * head_dim)^T = head_num * head_dim * embedding_dim
    half *w_v,    // head_num * (embedding_dim * head_dim)^T = head_num * head_dim * embedding_dim
    half *w_o,    // head_num * head_dim * embedding_dim
    half *k_cache,// batch * head_num * (seq_len - 1 (+1)) * head_dim
    half *v_cache,// batch * head_num * head_dim * (seq_len - 1 (+1))
    half *w_rms1, // embedding_dim
    half sm_scale,
    half *global_reduction // batch * head_num * embedding_dim
) {
    const float eps = 1e-5;

    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t head_idx = blockIdx.y;
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; // 32 per warp
    const uint32_t warp_id = tid / WARP_SIZE;

    const int EMBEDDING_TILE_LENGTH = EMBEDDING_DIM / ATTN_BLOCK_SIZE;
    const int ATTN_TILE_LENGTH = HEAD_DIM / (ATTN_BLOCK_SIZE / WARP_SIZE);

    // preallocate SRAM. (be careful with SRAM usage)
    __shared__ __align__(16) float block_reduce_float[ATTN_BLOCK_SIZE / WARP_SIZE];
    __shared__ __align__(16) half block_reduce_half[BLOCK_REDUCE_HALF];
    __shared__ __align__(16) half block_reduce_v[HEAD_DIM * ATTN_BLOCK_SIZE / WARP_SIZE]; 
    __shared__ __align__(16) half shared_HEAD_DIM0[HEAD_DIM], shared_HEAD_DIM1[HEAD_DIM];
    __shared__ __align__(16) half shared_residual[EMBEDDING_DIM];

    __align__(16) half weight_reg[EMBEDDING_TILE_LENGTH];
    __align__(16) half embedding_reg[EMBEDDING_TILE_LENGTH];
    __align__(16) half local_half1[ATTN_BLOCK_SIZE / WARP_SIZE], local_half2[ATTN_BLOCK_SIZE / WARP_SIZE]; // >= 8
    
    // #------------------------------------------------------------------------
    // # RMS Norm
    // #------------------------------------------------------------------------
    {
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_LENGTH / 8; di++) {
            *(uint4 * )(&shared_residual[tid * EMBEDDING_TILE_LENGTH + 8 * di]) = *(uint4 * )(&input[batch_idx * EMBEDDING_DIM + tid * EMBEDDING_TILE_LENGTH + 8 * di]);
            *(uint4 * )(&embedding_reg[8 * di]) = *(uint4 * )(&shared_residual[tid * EMBEDDING_TILE_LENGTH + 8 * di]);
        }
        float local_tmp_float = 0;
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_LENGTH; di++) {
            local_tmp_float += __half2float(embedding_reg[di]) * __half2float(embedding_reg[di]);
        }
        // reduce in block
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            local_tmp_float += __shfl_down_sync(0xffffffff, local_tmp_float, stride);
        }
        if (lane_id == 0) {
            block_reduce_float[warp_id] = local_tmp_float;
        }
        __syncthreads();
        local_tmp_float = 0;
        if (lane_id < ATTN_BLOCK_SIZE / WARP_SIZE) {
            local_tmp_float = block_reduce_float[lane_id];
        }
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            local_tmp_float += __shfl_down_sync(0xffffffff, local_tmp_float, stride);
        }
        // broad cast
        local_tmp_float = __shfl_sync(0xffffffff, local_tmp_float, 0);

        local_tmp_float =  __float2half(1.f / (std::sqrt(local_tmp_float / float(EMBEDDING_DIM))) + eps);
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_LENGTH / 8; di++) {
            *(uint4 * )(&weight_reg[8 * di]) = *(uint4 * )(&w_rms1[tid * EMBEDDING_TILE_LENGTH + 8 * di]);
        }
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_LENGTH; di++) {
            embedding_reg[di] = __hmul(__hmul(embedding_reg[di], local_tmp_float), weight_reg[di]);
        }
    }
    // #------------------------------------------------------------------------

    // #------------------------------------------------------------------------
    // # W_q x
    // #------------------------------------------------------------------------
    for (int i = 0; i < HEAD_DIM; i++) {
        half local_tmp_half = 0;
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_LENGTH / 8; di++) {
            *(uint4 * )(&weight_reg[8 * di]) = *(uint4 * )(&w_q[head_idx * EMBEDDING_DIM * HEAD_DIM + i * EMBEDDING_DIM + tid * EMBEDDING_TILE_LENGTH + 8 * di]);
        }
        for (int j = 0; j < EMBEDDING_TILE_LENGTH; ++j) {
            local_tmp_half += __hmul(embedding_reg[j], weight_reg[j]);
        }
        // warp level reduce
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            local_tmp_half +=  __shfl_down_sync(0xffffffff, local_tmp_half, stride);
        }
        if (lane_id == 0) {
            block_reduce_half[i * (ATTN_BLOCK_SIZE / WARP_SIZE) + warp_id] = local_tmp_half;
        }
    }

    // #------------------------------------------------------------------------
    // # W_k x 
    // #------------------------------------------------------------------------
    for (int i = 0; i < HEAD_DIM; i++) {
        half local_tmp_half = 0;
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_LENGTH / 8; di++) {
            *(uint4 * )(&weight_reg[8 * di]) = *(uint4 * )(&w_k[head_idx * EMBEDDING_DIM * HEAD_DIM + i * EMBEDDING_DIM + tid * EMBEDDING_TILE_LENGTH + 8 * di]);
        }
        for (int j = 0; j < EMBEDDING_TILE_LENGTH; ++j) {
            local_tmp_half += __hmul(embedding_reg[j], weight_reg[j]);
        }
        // warp level reduce
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            local_tmp_half +=  __shfl_down_sync(0xffffffff, local_tmp_half, stride);
        }
        if (lane_id == 0) {
            block_reduce_half[HEAD_DIM * (ATTN_BLOCK_SIZE / WARP_SIZE) + i * (ATTN_BLOCK_SIZE / WARP_SIZE) + warp_id] = local_tmp_half;
        }
    }
    __syncthreads();

    // #------------------------------------------------------------------------
    // # rope 
    // # ~~~~~
    // # [0, HEAD_DIM / 2): rope Q
    // # [HEAD_DIM / 2, HEAD_DIM): rope K
    // #------------------------------------------------------------------------
    if (tid < HEAD_DIM) {
        int position = tid % (HEAD_DIM / 2);
        int type = tid / (HEAD_DIM / 2);
        float inv_freq = (ROPE_OFFSET / ROPE_SCALE) / (std::pow(ROPE_THETA, float(2 * position) / float(HEAD_DIM)));
        float cos = std::cos(inv_freq);
        float sin = std::sin(inv_freq);
        half local1 = 0, local2 = 0;
        #pragma unroll
        for (int di = 0; di < ATTN_BLOCK_SIZE / WARP_SIZE / 8; di++) {
            *(uint4 * )(&local_half1[8 * di]) = *(uint4 * )(&block_reduce_half[type * HEAD_DIM * (ATTN_BLOCK_SIZE / WARP_SIZE) + position * (ATTN_BLOCK_SIZE / WARP_SIZE) + 8 * di]);
        }
        #pragma unroll
        for (int di = 0; di < ATTN_BLOCK_SIZE / WARP_SIZE; di++) {
            local1 += local_half1[di];
        }
        #pragma unroll
        for (int di = 0; di < ATTN_BLOCK_SIZE / WARP_SIZE / 8; di++) {
            *(uint4 * )(&local_half1[8 * di]) = *(uint4 * )(&block_reduce_half[type * HEAD_DIM * (ATTN_BLOCK_SIZE / WARP_SIZE) + (position + HEAD_DIM / 2) * (ATTN_BLOCK_SIZE / WARP_SIZE) + 8 * di]);
        }
        #pragma unroll
        for (int di = 0; di < ATTN_BLOCK_SIZE / WARP_SIZE; di++) {
            local2 += local_half1[di];
        }
        if (type == 0) {
            shared_HEAD_DIM0[position] = __float2half(cos * __half2float(local1) - sin * __half2float(local2));
            shared_HEAD_DIM0[position + HEAD_DIM / 2] = __float2half(cos * __half2float(local2) + sin * __half2float(local1));
        } else {
            shared_HEAD_DIM1[position] = __float2half(cos * __half2float(local1) - sin * __half2float(local2));
            shared_HEAD_DIM1[position + HEAD_DIM / 2] = __float2half(cos * __half2float(local2) + sin * __half2float(local1));
        }
    }
    __syncthreads();
    // #------------------------------------------------------------------------

    // #------------------------------------------------------------------------
    // # W_v x
    // #------------------------------------------------------------------------
    for (int i = 0; i < HEAD_DIM; i++){
        half local_tmp_half = 0;
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_LENGTH / 8; di++) {
            *(uint4 * )(&weight_reg[8 * di]) = *(uint4 * )(&w_v[head_idx * EMBEDDING_DIM * HEAD_DIM + i * EMBEDDING_DIM + tid * EMBEDDING_TILE_LENGTH + 8 * di]);
        }
        for (int j = 0; j < EMBEDDING_TILE_LENGTH; ++j) {
            local_tmp_half += __hmul(embedding_reg[j], weight_reg[j]);
        }
        // warp level reduce
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            local_tmp_half +=  __shfl_down_sync(0xffffffff, local_tmp_half, stride);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
        }
        if (lane_id == 0) {
            block_reduce_v[i * (ATTN_BLOCK_SIZE / WARP_SIZE) + warp_id] = local_tmp_half;
        }
    }
    __syncthreads();
    // #------------------------------------------------------------------------
    
    // #------------------------------------------------------------------------
    // # Attention
    // # ~~~~~~~~~~~~~~~~~
    // # Q: shared_HEAD_DIM0
    // # K: shared_HEAD_DIM1
    // # V: block_reduce_v
    // #------------------------------------------------------------------------
    // load q from shared memory to local
    *(uint4 * )(&local_half2[0]) = *(uint4 * )(&shared_HEAD_DIM0[(lane_id % 16) * 8]);
    float prev_l = 0, current_l;
    float prev_m = -100, current_m;
    __align__(16) float local_float[ATTN_BLOCK_SIZE / WARP_SIZE];
    __align__(16) half local_attn_score[ATTN_SPLIT / WARP_SIZE]; // >= 8
    __align__(16) half local_attn[ATTN_TILE_LENGTH];
    // prev o: embedding_reg
    for (int iter = tid; iter < SEQ_LEN / 8; iter += ATTN_BLOCK_SIZE) {
        // attention score
        float local_tmp_float = 0; // every warp has a max
        for (int d_iter = 0; d_iter < ATTN_SPLIT / (ATTN_BLOCK_SIZE / WARP_SIZE * 8); d_iter += ATTN_BLOCK_SIZE) {
            #pragma unroll
            for (int di = 0; di < 8; di++) {
                local_attn_score[di] = __float2half(0.0);
            }
            for (int inner = 0; inner < 4; ++ inner) {
                int inner_id = inner * 2 + int(lane_id / 16);
                int line_id = (iter + d_iter) * 8 + inner_id;
                if (line_id == SEQ_LEN - 1){
                    *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&shared_HEAD_DIM1[8 * (lane_id % 16)]);
                } else {
                    *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&k_cache[
                        batch_idx * HEAD_NUM * SEQ_LEN * HEAD_DIM + head_idx * SEQ_LEN * HEAD_DIM + line_id * HEAD_DIM + (lane_id % 16) * 8
                    ]);
                }
                #pragma unroll
                for (int di = 0; di < 8; di++) {
                    local_attn_score[inner_id] += __half2float(local_half2[di]) * __half2float(weight_reg[di]);
                }
            }
            #pragma unroll
            for (int stride = 8; stride > 0; stride >>= 1) {
                #pragma unroll
                for (int di = 0; di < 8; di++){
                    local_attn_score[di] += __shfl_down_sync(0xffffffff, local_attn_score[di], stride);
                }
            }
            // broad cast
            #pragma unroll
            for (int di = 0; di < 8; di++){
                local_attn_score[di] = __shfl_sync(0xffffffff, local_attn_score[di], 0);
                local_tmp_float = max(local_tmp_float, local_attn_score[di]);
            }
            if (lane_id == 0) {
                #pragma unroll
                for (int di = 0; di < 8; di++){
                    local_attn[di] = __hdiv(local_attn_score[di], sm_scale);
                }
                *(uint4 * )(&block_reduce_half[d_iter * (ATTN_BLOCK_SIZE / WARP_SIZE * 8) + warp_id * 8]) = *(uint4 * )(&local_attn[0]);
            }
        }
        // rowmax warp level reduce
        if (lane_id == 0) {
            block_reduce_float[warp_id] = local_tmp_float;
        }
        __syncthreads();
        // rowmax
        current_m = prev_m;
        #pragma unroll
        for (int i = 0; i < (ATTN_BLOCK_SIZE / WARP_SIZE) / 4; ++i){
            *(uint4 * )(&local_float[i * 4]) = *(uint4 * )(&block_reduce_float[i * 4]);
        }
        #pragma unroll
        for (int i = 0; i < ATTN_BLOCK_SIZE / WARP_SIZE; ++i){
            current_m = max(current_m, local_float[i]);
        }
        // load score
        current_l = 0;
        #pragma unroll
        for (int i = 0; i < ATTN_SPLIT / (8 * WARP_SIZE); ++i) {
            *(uint4 * )(&local_attn_score[8 * i]) = *(uint4 * )(&block_reduce_half[lane_id * ATTN_SPLIT / WARP_SIZE + i * 8]);
            #pragma unroll
            for (int ki = 0; ki < 8; ki++) {
                float exp_ = exp(__half2float(local_attn_score[8 * i + ki]) - current_m); // safe softmax
                local_attn_score[8 * i + ki] = __float2half(exp_);
                current_l += exp_;
            }
        }
        // warp reduce to lane_id 0 
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            current_l += __shfl_down_sync(0xffffffff, current_l, stride);
        }
        // broad cast
        current_l = __shfl_sync(0xffffffff, current_l, 0);
        current_l += prev_l * exp(prev_m - current_m);
        // score * V
        // ATTN_TILE_LENGTH x (ATTN_SPLIT / WARP_SIZE)
        for (int i = 0; i < ATTN_TILE_LENGTH; ++i) {
            half attntion = 0;
            // starting column
            int column_id = (iter / ATTN_BLOCK_SIZE) * ATTN_SPLIT + lane_id * ATTN_SPLIT / WARP_SIZE;
            int row_id = warp_id * ATTN_TILE_LENGTH + i;
            for (int di = 0; di < ATTN_SPLIT / (WARP_SIZE * 8); ++di) { // 8
                *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&v_cache[
                    batch_idx * HEAD_NUM * SEQ_LEN * HEAD_DIM + 
                    head_idx * SEQ_LEN * HEAD_DIM + 
                    SEQ_LEN * row_id + column_id + 8 * di
                ]);
                if (column_id + 8 * di + 8 == SEQ_LEN) {
                    #pragma unroll
                    for (int ii = 0; ii < ATTN_BLOCK_SIZE / WARP_SIZE / 8; ii++) {
                        *(uint4 * )(&local_half1[8 * ii]) = *(uint4 * )(&block_reduce_v[row_id * ATTN_BLOCK_SIZE / WARP_SIZE + 8 * ii]);
                    }
                    half v_value = 0;
                    #pragma unroll
                    for (int ii = 0; ii < ATTN_BLOCK_SIZE / WARP_SIZE; ii++) {
                        v_value += local_half1[ii];
                    }
                    weight_reg[7] = v_value;
                }
                #pragma unroll
                for (int ki = 0; ki < 8; ki++) {
                    attntion += __hmul(local_attn_score[di * 8 + ki], weight_reg[ki]);
                }
            }
            local_attn[i] = attntion;
        }
        for (int i = 0; i < ATTN_TILE_LENGTH; ++i) {
            #pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                local_attn[i] += __shfl_down_sync(0xffffffff, local_attn[i], stride);
            }
            local_attn[i] = __shfl_sync(0xffffffff, local_attn[i], 0);
        }
        #pragma unroll
        for (int i = 0; i < ATTN_TILE_LENGTH; ++i) {
            embedding_reg[i] = __float2half(__half2float(embedding_reg[i]) * exp(prev_m - current_m) + __half2float(local_attn[i]));
        }
        prev_m = current_m;
        prev_l = current_l;
    }
    #pragma unroll
    for (int i = 0; i < ATTN_TILE_LENGTH; ++i) {
        embedding_reg[i] = __hdiv(embedding_reg[i], current_l);
    }
    __syncthreads();
    // #------------------------------------------------------------------------
    // now, for each warp, embedding_reg contains a tile of attn value

    // #------------------------------------------------------------------------
    // # Attention Projection
    // #------------------------------------------------------------------------
    for (int iter = 0; iter < EMBEDDING_DIM / (WARP_SIZE * 8); ++iter) {
        __align__(16) half attn_value[8] = {};
        for (int iter_inner = 0; iter_inner < ATTN_TILE_LENGTH; ++iter_inner) {
            *(uint4 * )(&local_half1[0]) = *(uint4 * )(&w_o[
                head_idx * EMBEDDING_DIM * HEAD_DIM + 
                (warp_id * ATTN_TILE_LENGTH + iter_inner) * EMBEDDING_DIM + 
                WARP_SIZE * 8 * iter + lane_id * 8
            ]);
            #pragma unroll
            for (int i = 0; i < 8; ++i){
                attn_value[i] += __hmul(embedding_reg[iter_inner], local_half1[i]);
            }
        }
        *(uint4 * )(&block_reduce_half[warp_id * WARP_SIZE * 8 + 8 * lane_id]) = *(uint4 * )(&attn_value[0]);
        __syncthreads();
        if (warp_id == 0) {
            __align__(16) half local_reduction[8] = {};
            for (int iter_inner = 0; iter_inner < ATTN_BLOCK_SIZE / WARP_SIZE; ++iter_inner) {
                *(uint4 * )(&attn_value[0]) = *(uint4 * )(&block_reduce_half[iter_inner * WARP_SIZE * 8 + 8 * lane_id]);
                #pragma unroll
                for (int i = 0; i < 8; ++i){
                    local_reduction[i] += attn_value[i];
                }
            }
            *(uint4 * )(&global_reduction[
                batch_idx * HEAD_NUM * EMBEDDING_DIM + head_idx * EMBEDDING_DIM + 
                (WARP_SIZE * 8) * iter + 8 * lane_id
            ]) = *(uint4 * )(&local_reduction[0]);
        }
    }
    grid.sync();
    // #------------------------------------------------------------------------
    // now, attention value of each head is written in global

    // #------------------------------------------------------------------------
    // # Residual
    // #------------------------------------------------------------------------
    if (head_idx == 0) {
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_LENGTH / 8; di++) {
            *(uint4 * )(&embedding_reg[8 * di]) = *(uint4 * )(&shared_residual[tid * EMBEDDING_TILE_LENGTH + 8 * di]);
        }
        #pragma unroll
        for (int i = 0; i < HEAD_NUM; ++i) {
            #pragma unroll
            for (int di = 0; di < EMBEDDING_TILE_LENGTH / 8; di++) {
                *(uint4 * )(&weight_reg[8 * di]) = *(uint4 * )(&global_reduction[
                    batch_idx * HEAD_NUM * EMBEDDING_DIM + 
                    i * EMBEDDING_DIM +
                    tid * EMBEDDING_TILE_LENGTH + 8 * di
                ]);
            }
            #pragma unroll
            for (int di = 0; di < EMBEDDING_TILE_LENGTH; ++di) {
                embedding_reg[di] += weight_reg[di];
            } 
        }
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_LENGTH  / 8; ++di) {
            *(uint4 * )(&output[batch_idx * EMBEDDING_DIM + tid * EMBEDDING_TILE_LENGTH + 8 * di]) = *(uint4 * )(&embedding_reg[8 * di]);
        }
    }
    // #------------------------------------------------------------------------
    // [NOTE] Latency: 0.34559   
}

#define WARP_GROUP_SIZE (EMBEDDING_DIM/WARP_SIZE/8)
#define FFN_GROUP_LENGTH 128
#define FFN_BLOCK_REDUCE_HALF MAX(EMBEDDING_DIM, 2*(FFN_BLOCK_SIZE / WARP_SIZE)*FFN_GROUP_LENGTH)

// I HATE GLOBAL REDUCTION!
__global__ void decode_part2(
    half *output, // batch * embedding_dim
    half *input,  // batch * 1 * embedding_dim
    half *ffn_1,  // ffn_hidden * embedding_dim
    half *ffn_2,  // embedding_dim * ffn_hidden
    half *ffn_3,  // ffn_hidden * embedding_dim
    half *w_rms2  // embedding_dim
) {
    const float eps = 1e-5;
  
    cg::thread_block block = cg::this_thread_block();
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; // 32 per warp
    const uint32_t warp_id = tid / WARP_SIZE;

    const uint32_t group_id = warp_id / WARP_GROUP_SIZE;
    const uint32_t tile_id = warp_id % WARP_GROUP_SIZE;

    __shared__ __align__(16) float block_reduce_float[FFN_BLOCK_SIZE / WARP_SIZE];
    __shared__ __align__(16) half shared_residual[EMBEDDING_DIM];
    __shared__ __align__(16) half block_reduce_half[FFN_BLOCK_REDUCE_HALF];
    __shared__ __align__(16) half shared_ffn[FFN_HIDDEN];

    __align__(16) half embedding_reg[8], weight_reg[8];

    // #------------------------------------------------------------------------
    // # RMS Norm
    // #------------------------------------------------------------------------
    {
        float local_tmp_float = 0;
        if (group_id == 0){
            *(uint4 * )(&shared_residual[tile_id * 8]) = *(uint4 * )(&input[batch_idx * EMBEDDING_DIM + tile_id * 8]);
            *(uint4 * )(&block_reduce_half[tile_id * 8]) = *(uint4 * )(&w_rms2[batch_idx * EMBEDDING_DIM + tile_id * 8]);
            *(uint4 * )(&embedding_reg[0]) = *(uint4 * )(&shared_residual[tile_id * 8]);
            #pragma unroll
            for (int di = 0; di < 8; di++) {
                local_tmp_float += __half2float(embedding_reg[di]) * __half2float(embedding_reg[di]);
            }
        };
        // reduce in block
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            local_tmp_float += __shfl_down_sync(0xffffffff, local_tmp_float, stride);
        }
        if (group_id == 0 && lane_id == 0) {
            block_reduce_float[warp_id] = local_tmp_float;
        }
        __syncthreads();
        local_tmp_float = 0;
        if (lane_id < EMBEDDING_DIM / WARP_SIZE / 8) {
            local_tmp_float = block_reduce_float[lane_id];
        }
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            local_tmp_float += __shfl_down_sync(0xffffffff, local_tmp_float, stride);
        }
        // broad cast
        local_tmp_float = __shfl_sync(0xffffffff, local_tmp_float, 0);

        local_tmp_float =  __float2half(1.f / (std::sqrt(local_tmp_float / float(EMBEDDING_DIM))) + eps);
        *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&block_reduce_half[tile_id * 8]);
        if (group_id > 0) {
            *(uint4 * )(&embedding_reg[0]) = *(uint4 * )(&shared_residual[tile_id * 8]);
        }
        #pragma unroll
        for (int di = 0; di < 8; di++) {
            embedding_reg[di] = __hmul(__hmul(embedding_reg[di], local_tmp_float), weight_reg[di]);
        }
    }

    // #------------------------------------------------------------------------
    // # FFN
    // #------------------------------------------------------------------------
    // (ffn 1 + silu) * ffn 3
    for (int iter = 0; iter < (FFN_HIDDEN / (FFN_BLOCK_SIZE / WARP_SIZE / WARP_GROUP_SIZE)) / FFN_GROUP_LENGTH; iter++){ 
        for (int i = 0; i < FFN_GROUP_LENGTH; i++) {
            float thread_sum_ffn1 = 0, thread_sum_ffn3 = 0;
            // ffn 1
            *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&ffn_1[
                (group_id * (FFN_HIDDEN / (FFN_BLOCK_SIZE / WARP_SIZE / WARP_GROUP_SIZE)) + iter * FFN_GROUP_LENGTH + i) * EMBEDDING_DIM 
                + tile_id * 8
            ]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                thread_sum_ffn1 += __half2float(embedding_reg[j]) * __half2float(weight_reg[j]);
            }
            // ffn 3
            *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&ffn_3[
                (group_id * (FFN_HIDDEN / (FFN_BLOCK_SIZE / WARP_SIZE / WARP_GROUP_SIZE)) + iter * FFN_GROUP_LENGTH + i) * EMBEDDING_DIM 
                + tile_id * 8
            ]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                thread_sum_ffn3 += __half2float(embedding_reg[j]) * __half2float(weight_reg[j]);
            }
            // warp level reduce
            #pragma unroll
            for (int stride = WARP_GROUP_SIZE>>1; stride > 0; stride >>= 1) {
                thread_sum_ffn1 += __shfl_down_sync(0xffffffff, thread_sum_ffn1, stride);
                thread_sum_ffn3 += __shfl_down_sync(0xffffffff, thread_sum_ffn3, stride);
            }
            if (lane_id == 0) {
                block_reduce_half[i * (FFN_BLOCK_SIZE / WARP_SIZE) + warp_id] = thread_sum_ffn1;
                block_reduce_half[(FFN_GROUP_LENGTH + i) * (FFN_BLOCK_SIZE / WARP_SIZE) + warp_id] = thread_sum_ffn3;
            }
        }
        __syncthreads();

        __align__(16) half local_half[WARP_GROUP_SIZE];
        if (tile_id < FFN_GROUP_LENGTH) {
            half local1 = 0, local2 = 0;
            #pragma unroll
            for (int di = 0; di < WARP_GROUP_SIZE / 8; di++) {
                *(uint4 * )(&local_half[8 * di]) = *(uint4 * )(&block_reduce_half[
                    tid * (FFN_BLOCK_SIZE / WARP_SIZE) + 8 * di
                ]);
            }
            #pragma unroll
            for (int di = 0; di < WARP_GROUP_SIZE; di++) {
                local1 += local_half[di];
            }
            #pragma unroll
            for (int di = 0; di < WARP_GROUP_SIZE / 8; di++) {
                *(uint4 * )(&local_half[8 * di]) = *(uint4 * )(&block_reduce_half[
                    (FFN_GROUP_LENGTH + tid) * (FFN_BLOCK_SIZE / WARP_SIZE) + 8 * di
                ]);
            }
            #pragma unroll
            for (int di = 0; di < WARP_GROUP_SIZE; di++) {
                local2 += local_half[di];
            }
            float tmp = __half2float(local1);
            tmp /= (1.0f + expf(-tmp));
            shared_ffn[
                group_id * (FFN_HIDDEN / (FFN_BLOCK_SIZE / WARP_SIZE / WARP_GROUP_SIZE)) + iter * FFN_GROUP_LENGTH + tile_id
            ] = __hmul(tmp, local2);
        }
    }
    __syncthreads();
    // ffn 2 + residual
    for (int i = 0; i < EMBEDDING_DIM / (FFN_BLOCK_SIZE / WARP_SIZE); i++) {
        half sum = 0;
        for (int di = 0; di < FFN_HIDDEN/WARP_SIZE/8; ++di){
            *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&ffn_2[
                (warp_id * EMBEDDING_DIM / (FFN_BLOCK_SIZE / WARP_SIZE) + i) * FFN_HIDDEN +
                di * FFN_HIDDEN/WARP_SIZE + 8 * lane_id
            ]);
            #pragma unroll
            for (int ii = 0; ii < 8; ii++) {
                sum += __hmul(weight_reg[ii], shared_ffn[di * FFN_HIDDEN/WARP_SIZE + 8 * lane_id + ii]);
            }
        }
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, stride);
        }
        if(lane_id == 0){
            output[
                batch_idx *EMBEDDING_DIM + warp_id * EMBEDDING_DIM / (FFN_BLOCK_SIZE / WARP_SIZE) + i
            ] = shared_residual[warp_id * EMBEDDING_DIM / (FFN_BLOCK_SIZE / WARP_SIZE) + i] + sum;
        }
    }
}

int main(int argc, char **argv) {
    // shared memory size per threadBlock
    size_t dynamicShMemSize = sizeof(float) * (ATTN_BLOCK_SIZE / WARP_SIZE) + sizeof(half) * (EMBEDDING_DIM + BLOCK_REDUCE_HALF + HEAD_DIM * (2 + ATTN_BLOCK_SIZE / WARP_SIZE));
    cudaFuncSetAttribute(decode_part1, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicShMemSize);

    half sm_scale = __float2half(sqrt(HEAD_DIM));
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
    
    void *part1_kernelArgs[] = {
            &d_output, &d_input,
            &d_w_q, &d_w_k, &d_w_v, &d_w_o,
            &d_k_cache, &d_v_cache,
            &d_rms_1,
            &sm_scale,
            &global_reduce
    };

    void *part2_kernelArgs[] = {
        &d_input, &d_output,
        &d_ffn_1, &d_ffn_2, &d_ffn_3,
        &d_rms_2
    };

    // int minGridSize, blockSize;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, decode_part1, dynamicShMemSize);
    // printf("Optimal Block Size: %d\n", blockSize);

    /**
     * ! Without cluster, we have to do heavy global sync in grid scope.
     *      So we try to do less intro-block sync in the kernel.
     */
    dim3 part1_grid(BATCH_SIZE, HEAD_NUM, COOPERATE_BLOCK_NUM/HEAD_NUM);
    dim3 part1_block(ATTN_BLOCK_SIZE);
    dim3 part2_grid(BATCH_SIZE, 1, 1);
    dim3 part2_block(FFN_BLOCK_SIZE);

    int warmup = 50;
    int test = 100;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);

    // * warm up
    for (int i = 0; i < warmup; i++) {
        cudaLaunchCooperativeKernel((void *) decode_part1, part1_grid, part1_block, part1_kernelArgs, dynamicShMemSize);
        cudaLaunchCooperativeKernel((void *) decode_part2, part2_grid, part2_block, part2_kernelArgs, dynamicShMemSize);
    }

    // * test kernel
    cudaEventRecord(st);
    for (int i = 0; i < test; i++) {
        cudaLaunchCooperativeKernel((void *) decode_part1, part1_grid, part1_block, part1_kernelArgs, dynamicShMemSize);
        cudaLaunchCooperativeKernel((void *) decode_part2, part2_grid, part2_block, part2_kernelArgs, dynamicShMemSize);
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);

    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << (ms / (1.0 * test)) << " ms" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void *>(d_input), sizeof(half) * BATCH_SIZE * EMBEDDING_DIM,
               cudaMemcpyDeviceToHost);

    return 0;
}