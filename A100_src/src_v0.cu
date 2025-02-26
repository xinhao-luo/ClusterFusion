#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>
#include <math.h>

namespace cg = cooperative_groups;

// nvcc -arch=sm_80 -std=c++17 fused_decode.cu -o test && ./test

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
#define HEAD_DIM 1024        // attn head dimension
#define HEAD_NUM 8      // attn head number
#define SMALL_COOPERATE_BLOCK_NUM (COOPERATE_BLOCK_NUM/HEAD_NUM) // 16

#define FFN_HIDDEN 4096     // ffn hidden dimension
#define EMBEDDING_DIM 4096  // token embedding dimension
#define SEQ_LEN 4096        // sequence length
#define FL_DEC_SPLIT 256 // FL_DEC_SPLIT <= SEQ_LEN

#define WB_WORKLOAD  8*((HEAD_DIM+8*SMALL_COOPERATE_BLOCK_NUM-1)/(8*SMALL_COOPERATE_BLOCK_NUM))
#define ATTN_WORKLOAD FL_DEC_SPLIT*((SEQ_LEN+(FL_DEC_SPLIT*SMALL_COOPERATE_BLOCK_NUM)-1)/(FL_DEC_SPLIT*SMALL_COOPERATE_BLOCK_NUM))

#if  SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM > EMBEDDING_DIM
#define ATTN_REDUCE_SIZE BATCH_SIZE * HEAD_NUM * (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM)
#else
#define ATTN_REDUCE_SIZE BATCH_SIZE * HEAD_NUM *  EMBEDDING_DIM
#endif

std::mt19937 rng(42);
std::normal_distribution<float> norm_dist(0.0, 1.0);

template<typename T>
void fill_matrix(T *mat, int sz) {
    for (int i = 0; i < sz; i++) {
        float random_value;
        do {
            random_value = norm_dist(rng);
        } while (random_value < -1 || random_value > 1);
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
 * @param output ptr -> output embedding in global
 * @param embedding ptr -> input embedding in global
 * @param w_rms ptr -> global
 * @param block_reduce_mem ptr -> SRAM
 */
__device__ __forceinline__ void RMSNorm_global(
        half *output,
        half *embedding, half *batch_reduce,
        half *w_rms,
        float *block_reduce_mem,
        float eps = 1e-5
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t head_idx = blockIdx.y;
    const uint32_t cooperate_idx = blockIdx.z;
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % 32; // 32 per warp
    const uint32_t warp_id = tid / 32;
    int d = head_idx * BLOCK_SIZE * SMALL_COOPERATE_BLOCK_NUM + BLOCK_SIZE * cooperate_idx + tid;

    __align__(16)
    half w_rms_reg[8], embedding_reg[8], tmp_reg[8];
    float thread_sum = 0;
    if (d < EMBEDDING_DIM / 8) {
        *(uint4 * )(&embedding_reg[0]) = *(uint4 * )(&embedding[batch_idx * EMBEDDING_DIM + d * 8]);

        for (int di = 0; di < 8; di++) {
            thread_sum += __half2float(embedding_reg[di]) * __half2float(embedding_reg[di]);
        }
    }
    // reduce in block
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, mask);
    }

    if (lane_id == 0) {
        block_reduce_mem[warp_id] = thread_sum;
    }
    __syncthreads();
    if (tid == 0) {
        thread_sum = 0;
#pragma unroll
        for (int di = 0; di < BLOCK_SIZE / 32; di++) {
            thread_sum += block_reduce_mem[di];
        }
        atomicAdd(&batch_reduce[batch_idx], thread_sum);
    }
    grid.sync();
    if (d < EMBEDDING_DIM / 8) {
        half rms_rcp = __float2half(
                1.f / (std::sqrt(__half2float(batch_reduce[batch_idx]) / float(EMBEDDING_DIM))) + eps);
        *(uint4 * )(&w_rms_reg[0]) = *(uint4 * )(&w_rms[d * 8]);
        for (int di = 0; di < 8; di++) {
            tmp_reg[di] = __hmul(__hmul(embedding_reg[di], rms_rcp), w_rms_reg[di]);
        }
        *(uint4 * )(&output[batch_idx * EMBEDDING_DIM + d * 8]) = *(uint4 * )(tmp_reg);
    }
    grid.sync();
}

/**
 *
 * @param output ptr -> block shared memory
 * @param embedding -> input embedding in global
 * @param w_rms ptr -> global
 * @param block_reduce_mem ptr -> SRAM
 */
__device__ __forceinline__ void RMSNorm_shmem(
        half *output,
        half *embedding,
        half *w_rms,
        float *block_reduce_mem,
        float eps = 1e-5
) {
    cg::thread_block block = cg::this_thread_block();
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % 32; // 32 per warp
    const uint32_t warp_id = tid / 32;

    __align__(16)
    half w_rms_reg[8], embedding_reg[8], tmp_reg[8];
    float thread_sum = 0;
    for (int i = tid; i < EMBEDDING_DIM / 8; i += BLOCK_SIZE) {
        *(uint4 * )(&embedding_reg[0]) = *(uint4 * )(&embedding[batch_idx * EMBEDDING_DIM + i * 8]);
        for (int di = 0; di < 8; di++) {
            thread_sum += __half2float(embedding_reg[di]) * __half2float(embedding_reg[di]);
        }
    }
    // reduce in block
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, mask);
    }

    if (lane_id == 0) {
        block_reduce_mem[warp_id] = thread_sum;
    }
    __syncthreads();
    if (tid == 0) {
        thread_sum = 0;
#pragma unroll
        for (int di = 0; di < BLOCK_SIZE / 32; di++) {
            thread_sum += block_reduce_mem[di];
        }
        block_reduce_mem[0] = thread_sum;
    }
    half rms_rcp = __float2half(1.f / (std::sqrt(block_reduce_mem[0] / float(EMBEDDING_DIM))) + eps);
    for (int i = tid; i < EMBEDDING_DIM / 8; i += BLOCK_SIZE) {
        *(uint4 * )(&w_rms_reg[0]) = *(uint4 * )(&w_rms[i * 8]);
        for (int di = 0; di < 8; di++) {
            tmp_reg[di] = __hmul(__hmul(embedding_reg[di], rms_rcp), w_rms_reg[di]);
        }
        *(uint4 * )(&output[i * 8]) = *(uint4 * )(tmp_reg);
    }
}

/**
 *
 * @param output ptr -> global
 * @param x [1 x 8] ptr -> register
 * @param W ptr -> global weight matrix
 * @param width
 * |    - qkv: EMBEDDING_DIM
 * |    - output: FFN_HIDDEN
 * @param length xW is [1 x length]
 * |    - qkv: HEAD_DIM
 * |    - output: EMBEDDING_DIM
 * @param weight_offset
 * |    - qkv: head_idx * EMBEDDING_DIM * HEAD_DIM + cooperative_id * 8
 * |    - output: i * 8
 * @param output_offset
 * |    - qkv: batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM
 * |    - output: batch_idx * EMBEDDING_DIM
 */
__device__ __forceinline__ void xW_global(
        half *output, half *block_reduce_mem,
        half *x, half *W, int width,
        int length, int weight_offset, int output_offset,
        bool active
) {
    const uint32_t tid = cg::this_thread_block().thread_rank();
    const uint32_t lane_id = tid % 32; // 32 per warp
    const uint32_t warp_id = tid / 32;

    float mul_sum;
    __align__(16)
    half weight_reg[8];

    for (int i = 0; i < length; i++) {
        if (lane_id == 0) {
            block_reduce_mem[warp_id] = 0;
        }
        __syncthreads();

        mul_sum = 0;
        if (active) {
            *(uint4 * )(&weight_reg[0]) = *(uint4 * )(&W[weight_offset + i * width]);
            for (int j = 0; j < 8; ++j) {
                mul_sum += __half2float(x[j]) * __half2float(weight_reg[j]);
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            mul_sum += __shfl_down_sync(0xffffffff, mul_sum, mask);
        }
        if (active && lane_id == 0) {
            block_reduce_mem[warp_id] += mul_sum;
        }

        __syncthreads();
        if (active && tid == 0) {
            half sum = 0;
#pragma unroll
            for (int di = 0; di < BLOCK_SIZE / 32; di++) {
                sum += block_reduce_mem[di];
            }
            atomicAdd(&output[output_offset + i], sum);
        }
    }
}


__device__ __forceinline__ void rope(
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
        *(uint4 * )(&output[d * 8])
                = *(uint4 * )(roped_output_reg);
    }
}

/**
 *
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
 * @param global_q
 * @param global_reduce_tmp_qkv
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
        half *ffn_1,  // (embedding_dim * ffn_hidden)
        half *ffn_2,  // (ffn_hidden * embedding_dim)^T
        half *ffn_3,  // (embedding_dim * ffn_hidden)
        half *w_rms1, // embedding_dim
        half *w_rms2, // embedding_dim
        int rope_offset, // offset of RoPE starting point
        float rope_scale,
        float rope_theta,
        float sm_scale,
        half *global_q, // batch * head_num * head_dim
        half *global_batch_reduction, // batch
        half *global_reduce_tmp_qkv, // 3 * batch * head_num * head_dim
        half *global_attn_reduction, // batch * head_num * (SMALL_COOPERATE_BLOCK_NUM * head_dim)
        float *global_attn_scale // batch * head_num * SMALL_COOPERATE_BLOCK_NUM
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t head_idx = blockIdx.y;
    const uint32_t cooperate_idx = blockIdx.z;
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % 32; // 32 per warp
    const uint32_t warp_id = tid / 32;

    int cooperative_id = BLOCK_SIZE * cooperate_idx + tid;
    int larger_cooperative_id =
            head_idx * BLOCK_SIZE * SMALL_COOPERATE_BLOCK_NUM + BLOCK_SIZE * cooperate_idx + tid;
    // init global reduce space with 0
    uint4 zero_initializer = {0, 0, 0, 0};
    if (cooperative_id < HEAD_DIM / 8) {
        // qkv
        // todo: can be optimized
        *(uint4 * )(&global_reduce_tmp_qkv[
                batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM + cooperative_id * 8
        ]) = zero_initializer;
        *(uint4 * )(&global_reduce_tmp_qkv[
                (batch_idx + BATCH_SIZE) * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM + cooperative_id * 8
        ]) = zero_initializer;
        *(uint4 * )(&global_reduce_tmp_qkv[
                (batch_idx + 2 * BATCH_SIZE) * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM + cooperative_id * 8
        ]) = zero_initializer;
    }
    if (larger_cooperative_id == 0) {
        global_batch_reduction[batch_idx] = 0;
    }
    grid.sync();

    // preallocate SRAM. (be careful with SRAM usage)
    __shared__ float block_reduce_float[BLOCK_SIZE / 32];
    __shared__ float tmp_mem[8];
    __shared__ __align__
    (16)
    half local_HEAD_DIM[HEAD_DIM], local_HEAD_DIM_prev[HEAD_DIM], local_EMBEDDING[EMBEDDING_DIM],
            block_reduce_half[BLOCK_SIZE / 32];
    __shared__ __align__
    (16)
    half local_Q[HEAD_DIM], local_FL_DEC_SPLIT[FL_DEC_SPLIT];
    __shared__ __align__
    (16)
    float local_slice_scale[SMALL_COOPERATE_BLOCK_NUM];

    /**
     * Prepare QKV
     */
    __align__(16)
    half embedding_reg[8];
    RMSNorm_global(output, input, global_batch_reduction, w_rms1, block_reduce_float);
    if (cooperative_id < EMBEDDING_DIM / 8) {
        *(uint4 * )(&embedding_reg[0]) = *(uint4 * )(&output[batch_idx * EMBEDDING_DIM + cooperative_id * 8]);
    }
    // calculate q, k, v
    xW_global(
            global_reduce_tmp_qkv, block_reduce_half, embedding_reg, w_q, EMBEDDING_DIM,
            HEAD_DIM, head_idx * EMBEDDING_DIM * HEAD_DIM + cooperative_id * 8,
            batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM, cooperative_id < EMBEDDING_DIM / 8
    ); // tmp q
    xW_global(
            &global_reduce_tmp_qkv[BATCH_SIZE * HEAD_NUM * HEAD_DIM],
            block_reduce_half, embedding_reg, w_k, EMBEDDING_DIM, HEAD_DIM,
            head_idx * EMBEDDING_DIM * HEAD_DIM + cooperative_id * 8,
            batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM, cooperative_id < EMBEDDING_DIM / 8
    ); // tmp k
    xW_global(
            &global_reduce_tmp_qkv[2 * BATCH_SIZE * HEAD_NUM * HEAD_DIM],
            block_reduce_half, embedding_reg, w_v, EMBEDDING_DIM, HEAD_DIM,
            head_idx * EMBEDDING_DIM * HEAD_DIM + cooperative_id * 8,
            batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM, cooperative_id < EMBEDDING_DIM / 8
    );
    grid.sync();
    // rope on q, k
    rope(&global_q[batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM], global_reduce_tmp_qkv, rope_offset,
         rope_scale, rope_theta);
    rope(&k_cache[batch_idx * HEAD_NUM * SEQ_LEN * HEAD_DIM + head_idx * SEQ_LEN * HEAD_DIM + (SEQ_LEN - 1) * HEAD_DIM],
         &global_reduce_tmp_qkv[BATCH_SIZE * HEAD_NUM * HEAD_DIM], rope_offset, rope_scale, rope_theta);

    __align__(16)
    half kv_cache_reg[8];
    if (tid == 0) {
        for (int i = cooperate_idx * WB_WORKLOAD; i < HEAD_DIM; i += 8) {
            *(uint4 * )(kv_cache_reg) = *(uint4 * )(&global_reduce_tmp_qkv[
                    2 * BATCH_SIZE * HEAD_NUM * HEAD_DIM + batch_idx * HEAD_NUM * HEAD_DIM
                    + head_idx * HEAD_DIM + i
            ]);
#pragma unroll
            for (int di = 0; di < 8; di++) {
                v_cache[
                        batch_idx * HEAD_NUM * HEAD_DIM * SEQ_LEN + head_idx * HEAD_DIM * SEQ_LEN
                        + (i + di) * SEQ_LEN + SEQ_LEN - 1
                ] = kv_cache_reg[di];
            }
        }
    }
    grid.sync();

    /**
     * Attention
     * ----------------------------------------
     * flash decoding
     * SMALL_COOPERATE_BLOCK_NUM blocks cooperate to split SEQ_LEN and reduce in HBM.
     * Each block tackle with SEQ_LEN / SMALL_COOPERATE_BLOCK_NUM using flash attention.
     * Every time, the block calculates FL_DEC_SPLIT.
     * todo: safe Softmax
     */
    // load Q [1 x HEAD_DIM] to share memory
    if (tid < HEAD_DIM / 8) {
        *(uint4 * )(&local_Q[tid * 8]) = *(uint4 * )(
                &global_q[batch_idx * HEAD_DIM * HEAD_NUM + head_idx * HEAD_DIM + tid * 8]);
    }
    __syncthreads();

    __align__(16)
    half local_q_reg[8], local_s_reg[8], local_o_reg[8], local_prev_o_reg[8];
    float local_scale = 0, local_prev_scale = 0;
    // init
    if (tid == 0) {
        tmp_mem[1] = 0;
        for (int di = 0; di < HEAD_DIM; ++di) {
            local_HEAD_DIM[di] = __float2half(0.0f);
            local_HEAD_DIM_prev[di] = __float2half(0.0f);
        }
    }
    int active_cooperator = SEQ_LEN >= (FL_DEC_SPLIT * SMALL_COOPERATE_BLOCK_NUM) ? SMALL_COOPERATE_BLOCK_NUM :
                            (SEQ_LEN / FL_DEC_SPLIT + SMALL_COOPERATE_BLOCK_NUM - 1) / SMALL_COOPERATE_BLOCK_NUM;

    // working iteration, one chunk per iteration
    for (int iter = 0; iter < ATTN_WORKLOAD / FL_DEC_SPLIT; iter++) {
        local_scale = 0; // chunk scale
        for (int j = tid; j < FL_DEC_SPLIT; j += BLOCK_SIZE) {
            int cache_idx = cooperate_idx * ATTN_WORKLOAD + iter * FL_DEC_SPLIT + j;
            if (cache_idx < SEQ_LEN) {
                half tmp_local_sum_reg = __float2half(0.0f);
                for (int di = 0; di < HEAD_DIM / 8; ++di) {
                    *(uint4 * )(kv_cache_reg) = *(uint4 * )(&k_cache[
                            batch_idx * HEAD_NUM * SEQ_LEN * HEAD_DIM + head_idx * SEQ_LEN * HEAD_DIM
                            + cache_idx * HEAD_DIM + 8 * di
                    ]);
                    *(uint4 * )(local_q_reg) = *(uint4 * )(&local_Q[di * 8]);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        tmp_local_sum_reg += __hmul(kv_cache_reg[i], local_q_reg[i]);
                    }
                }
                float tmp = exp(__half2float(tmp_local_sum_reg) / sm_scale);
                local_FL_DEC_SPLIT[j] = __float2half(tmp);
                local_scale += tmp;
            }
        }

        // reduce scale in warp
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_scale += __shfl_down_sync(0xffffffff, local_scale, mask);
        }
        if (lane_id == 0) {
            local_prev_scale += local_scale;
            block_reduce_float[warp_id] = local_prev_scale; // current iteration, current warp
        }
        __syncthreads();

        // v
        for (int j = tid; j < HEAD_DIM; j += BLOCK_SIZE) {
            half sum_reg = 0;
            for (int di = 0; di < FL_DEC_SPLIT / 8; ++di) {
                int cache_idx = cooperate_idx * ATTN_WORKLOAD + iter * FL_DEC_SPLIT + 8 * di;
                if (cache_idx < SEQ_LEN) {
                    *(uint4 * )(kv_cache_reg) = *(uint4 * )(&v_cache[
                            batch_idx * HEAD_NUM * SEQ_LEN * HEAD_DIM + head_idx * SEQ_LEN * HEAD_DIM
                            + j * SEQ_LEN + cache_idx
                    ]);
                    *(uint4 * )(local_s_reg) = *(uint4 * )(&local_FL_DEC_SPLIT[di * 8]);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        sum_reg += __hmul(kv_cache_reg[i], local_s_reg[i]);
                    }
                }
            }
            local_HEAD_DIM[j] = sum_reg;
        }
        __syncthreads();

        if (tid == 0) {
            for (int i = 1; i < BLOCK_SIZE / 32; i++) {
                block_reduce_float[0] += block_reduce_float[i];
            }
            tmp_mem[0] = tmp_mem[1];// prev_attn_scale
            tmp_mem[1] = block_reduce_float[0];// cur_attn_scale
        }
        __syncthreads();
        if (tid < HEAD_DIM / 8) {
            *(uint4 * )(local_o_reg) = *(uint4 * )(&local_HEAD_DIM[tid * 8]);
            *(uint4 * )(local_prev_o_reg) = *(uint4 * )(&local_HEAD_DIM_prev[tid * 8]);
#pragma unroll
            for (int di = 0; di < 8; di++) {
                local_o_reg[di] = __float2half(
                        __half2float(local_prev_o_reg[di]) * tmp_mem[0] / tmp_mem[1] +
                        __half2float(local_o_reg[di]) / tmp_mem[1]
                );
            }
            *(uint4 * )(&local_HEAD_DIM_prev[tid * 8]) = *(uint4 * )(local_o_reg); // update
        }
        __syncthreads();
    }

    // put to global reduction
    if (cooperate_idx < active_cooperator) {
        if (tid < HEAD_DIM / 8) {
            *(uint4 * )(local_o_reg) = *(uint4 * )(&local_HEAD_DIM_prev[tid * 8]);
            // O
            *(uint4 * )(&global_attn_reduction[
                    batch_idx * HEAD_NUM * (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM)
                    + head_idx * (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM)
                    + cooperate_idx * HEAD_DIM + tid * 8
            ]) = *(uint4 *) local_o_reg;
        }
        if (tid == 0) {
            global_attn_scale[
                    batch_idx * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM
                    + head_idx * SMALL_COOPERATE_BLOCK_NUM + cooperate_idx
            ] = tmp_mem[1];
        }
    }
    grid.sync();

    // the first block
    // flash decoding, 'cluster' level reduction
    // note that some of the cooperators may not be active
    float local_scale_sum = 0;
    if (cooperate_idx == 0) {
        if (tid == 0) {
            int idx;
            for (idx = 0; idx + 8 < active_cooperator; idx += 8) {
                *(uint4 * )(&local_slice_scale[idx]) = *(uint4 * )(&global_attn_scale[
                        batch_idx * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM
                        + head_idx * SMALL_COOPERATE_BLOCK_NUM + idx
                ]);
            }
            for (; idx < active_cooperator; ++idx) {
                local_slice_scale[idx] = global_attn_scale[
                        batch_idx * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM
                        + head_idx * SMALL_COOPERATE_BLOCK_NUM + idx
                ];
            }
            float scale_sum = 0;
#pragma unroll
            for (int di = 0; di < active_cooperator; di++) {
                scale_sum += local_slice_scale[di];
            }
            block_reduce_float[0] = scale_sum;
        }
        __syncthreads();
        if (tid < HEAD_DIM / 8) {
            *(uint4 *) local_prev_o_reg = *(uint4 * )(&global_attn_reduction[
                    batch_idx * HEAD_NUM * (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM)
                    + head_idx * (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM) + tid * 8
            ]);
            local_scale = local_slice_scale[0];
            local_scale_sum = block_reduce_float[0];
#pragma unroll
            for (int di = 0; di < 8; di++) {
                local_prev_o_reg[di] = __float2half(__half2float(
                        local_prev_o_reg[di]) * local_scale / local_scale_sum
                );
            }
            for (int j = 1; j < active_cooperator; ++j) {
                *(uint4 *) local_o_reg = *(uint4 * )(&global_attn_reduction[
                        batch_idx * HEAD_NUM * (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM)
                        + head_idx * (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM) + j * HEAD_DIM + tid * 8
                ]);
                local_scale = local_slice_scale[j];
#pragma unroll
                for (int di = 0; di < 8; di++) {
                    local_prev_o_reg[di] += __float2half(__half2float(
                            local_o_reg[di]) * local_scale / local_scale_sum
                    );
                }
            }
            // reuse for attention output
            *(uint4 * )(&global_attn_reduction[
                    batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM + tid * 8
            ]) = *(uint4 * )(&local_prev_o_reg[0]);
        }
    }
    grid.sync();

    if (cooperative_id < HEAD_DIM / 8) {
        *(uint4 * )(&embedding_reg[0]) = *(uint4 * )(&global_attn_reduction[
                batch_idx * HEAD_NUM * HEAD_DIM + head_idx * HEAD_DIM + cooperative_id * 8
        ]);
    }
    // init
    if (cooperative_id < EMBEDDING_DIM / 8) {
        *(uint4 * )(&global_attn_reduction[
                batch_idx * HEAD_NUM * EMBEDDING_DIM + head_idx * EMBEDDING_DIM + cooperative_id * 8
        ]) = zero_initializer;
    }
    grid.sync();
    xW_global(
            global_attn_reduction, block_reduce_half, embedding_reg, w_o, HEAD_DIM, EMBEDDING_DIM,
            head_idx * HEAD_DIM * EMBEDDING_DIM + cooperative_id * 8,
            batch_idx * HEAD_NUM * EMBEDDING_DIM + head_idx * EMBEDDING_DIM, cooperative_id < HEAD_DIM / 8
    );
    grid.sync();
    // reduce batch output
    if (head_idx == 0 && cooperate_idx == 0) {
        half local_EMBEDDING_reg[EMBEDDING_DIM] = {__float2half(0.0),};
#pragma unroll
        for (int h = 0; h < HEAD_NUM; ++h) {
            for (int i = 0; i < EMBEDDING_DIM / 8; ++i) {
                *(uint4 * )(embedding_reg) = *(uint4 * )(&global_attn_reduction[
                        batch_idx * HEAD_NUM * EMBEDDING_DIM + h * EMBEDDING_DIM + 8 * i
                ]);
#pragma unroll
                for (int di = 0; di < 8; di++) {
                    local_EMBEDDING_reg[8 * i + di] += embedding_reg[di];
                }
            }
        }
        for (int i = 0; i < EMBEDDING_DIM; ++i) {
            input[batch_idx * EMBEDDING_DIM + i] += local_EMBEDDING_reg[i];
        }
    }
    // norm and write to shared memory
    RMSNorm_shmem(local_EMBEDDING, input, w_rms2, block_reduce_float);
    // prepare for residual
    if (larger_cooperative_id == 0 && tid == 0) {
        for (int i = 0; i < EMBEDDING_DIM / 8; ++i) {
            *(uint4 * )(&output[batch_idx * EMBEDDING_DIM + 8 * i]) = *(uint4 * )(&local_EMBEDDING[8 * i]);
        }
    }

    /**
     * FFN
     * ----------------------------------------
     * every block deal with FFN_HIDDEN/COOPERATE_BLOCK_NUM
     */
    __align__(16)
    half local_weight_reg[8];
    __align__(16)
    half ffn_hidden_reg1[8] = {__float2half(0.0),}, ffn_hidden_reg2[8] = {__float2half(0.0),};
    if (larger_cooperative_id < FFN_HIDDEN / 8) {
        for (int i = 0; i < EMBEDDING_DIM / 8; i++) {
            *(uint4 * )(embedding_reg) = *(uint4 * )(&local_EMBEDDING[8 * i]);
            for (int di = 0; di < 8; di++) {
                *(uint4 * )(local_weight_reg) = *(uint4 * )(
                        &ffn_1[FFN_HIDDEN * (8 * i + di) + 8 * larger_cooperative_id]);
                for (int dj = 0; dj < 8; dj++) {
                    ffn_hidden_reg1[dj] += __hmul(embedding_reg[di], local_weight_reg[dj]);
                }
                *(uint4 * )(local_weight_reg) = *(uint4 * )(
                        &ffn_3[FFN_HIDDEN * (8 * i + di) + 8 * larger_cooperative_id]);
                for (int dj = 0; dj < 8; dj++) {
                    ffn_hidden_reg2[dj] += __hmul(embedding_reg[di], local_weight_reg[dj]);
                }
            }
        }
        // elementwise silu and multiply
        for (int di = 0; di < 8; di++) {
            float tmp = __half2float(ffn_hidden_reg1[di]);
            tmp /= (1.0f + expf(-tmp));
            tmp *= __half2float(ffn_hidden_reg2[di]);
            ffn_hidden_reg1[di] = __float2half(tmp);
        }
    }
    // W2
    xW_global(
            output, block_reduce_half, ffn_hidden_reg1, ffn_2, FFN_HIDDEN,
            EMBEDDING_DIM, larger_cooperative_id * 8, batch_idx * EMBEDDING_DIM,
            larger_cooperative_id < FFN_HIDDEN / 8
    );
}

int main(int argc, char **argv) {
    // shared memory size per threadBlock
    size_t dynamicShMemSize =
            sizeof(float) * (BLOCK_SIZE + SMALL_COOPERATE_BLOCK_NUM) +
            sizeof(half) * (HEAD_DIM * 4 + FL_DEC_SPLIT + EMBEDDING_DIM);
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

    half h_global_q[BATCH_SIZE * HEAD_NUM * HEAD_DIM] = {__float2half(0.0)};
    half h_global_batch[BATCH_SIZE] = {__float2half(0.0)};
    half h_global_tmp_qkv[3 * BATCH_SIZE * HEAD_NUM * HEAD_DIM] = {__float2half(0.0)};
    half h_global_reduce_attn[ATTN_REDUCE_SIZE];
    float h_global_reduce_attn_scale[BATCH_SIZE * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM];
    half *global_reduce_q;
    half *global_reduce_batch;
    half *global_reduce_tmp_qkv;
    half *global_reduce_attn;
    float *global_reduce_attn_scale;
    cudaMalloc(reinterpret_cast<void **>(&global_reduce_q), sizeof(half) * BATCH_SIZE * HEAD_NUM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&global_reduce_batch), sizeof(half) * BATCH_SIZE);
    cudaMalloc(reinterpret_cast<void **>(&global_reduce_tmp_qkv), sizeof(half) * 3 * BATCH_SIZE * HEAD_NUM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&global_reduce_attn), sizeof(half) * BATCH_SIZE * HEAD_NUM *
                                                               (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM));
    cudaMalloc(reinterpret_cast<void **>(&global_reduce_attn_scale),
               sizeof(float) * BATCH_SIZE * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM);
    cudaMemcpy(reinterpret_cast<void *>(global_reduce_q), h_global_q,
               sizeof(half) * BATCH_SIZE * HEAD_NUM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(global_reduce_batch), h_global_batch,
               sizeof(half) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(global_reduce_tmp_qkv), h_global_tmp_qkv,
               sizeof(half) * 3 * BATCH_SIZE * HEAD_NUM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(global_reduce_attn), h_global_reduce_attn,
               sizeof(half) * ATTN_REDUCE_SIZE,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(global_reduce_attn_scale), h_global_reduce_attn_scale,
               sizeof(float) * BATCH_SIZE * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM, cudaMemcpyHostToDevice);

    void *kernelArgs[] = {
            &d_output, &d_input,
            &d_w_q, &d_w_k, &d_w_v, &d_w_o,
            &d_k_cache, &d_v_cache,
            &d_ffn_1, &d_ffn_2, &d_ffn_3,
            &d_rms_1, &d_rms_2,
            &rope_offset, &rope_scale, &rope_theta,
            &sm_scale,
            &global_reduce_q,
            &global_reduce_batch,
            &global_reduce_tmp_qkv,
            &global_reduce_attn,
            &global_reduce_attn_scale
    };

    /**
     * ! Without cluster, we have to do heavy global sync in grid scope.
     *      So we try to do less intro-block sync in the kernel.
     */
    dim3 grid(BATCH_SIZE, HEAD_NUM, SMALL_COOPERATE_BLOCK_NUM);
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
    std::cout << "Latency: " << (ms / (1.0 * test)) * 1e3 << " us" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void *>(d_output), sizeof(half) * BATCH_SIZE * EMBEDDING_DIM,
               cudaMemcpyDeviceToHost);

    return 0;
}