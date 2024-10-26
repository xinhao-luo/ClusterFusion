#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>
#include <math.h>

namespace cg = cooperative_groups;

// nvcc -arch=sm_80 -std=c++17 attn.cu -o test && ./test

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
#define COOPERATE_BLOCK_NUM 16 // number of blocks working on a request (>HEAD_NUM)
/**
 * Decode config
 */
#define BATCH_SIZE 1
#define HEAD_DIM 64        // attn head dimension
#define HEAD_NUM 4         // attn head number
#define SMALL_COOPERATE_BLOCK_NUM (COOPERATE_BLOCK_NUM/HEAD_NUM) // 16

#define FFN_HIDDEN 512     // ffn hidden dimension
#define EMBEDDING_DIM 64  // token embedding dimension
#define SEQ_LEN 1024        // sequence length
#define FL_DEC_SPLIT 128
#define WORKLOAD FL_DEC_SPLIT*((SEQ_LEN+(FL_DEC_SPLIT*SMALL_COOPERATE_BLOCK_NUM)-1)/(FL_DEC_SPLIT*SMALL_COOPERATE_BLOCK_NUM))

// #define HEAD_DIM 256        // attn head dimension
// #define HEAD_NUM 32         // attn head number
// #define FFN_HIDDEN 4096     // ffn hidden dimension
// #define EMBEDDING_DIM 4096  // token embedding dimension
// #define SEQ_LEN 4096        // sequence length
// #define FL_DEC_SPLIT 256

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

/**
* Attention
* ----------------------------------------
* flash decoding
* SMALL_COOPERATE_BLOCK_NUM blocks cooperate to split SEQ_LEN and reduce in HBM.
* Each block tackle with SEQ_LEN / SMALL_COOPERATE_BLOCK_NUM using flash attention.
* Every time, the block calculates FL_DEC_SPLIT.
* todo: safe Softmax
*/
__global__ void attn(
        half *output, // batch * embedding_dim
        half *q,  // batch * head_num * head_dim
        half *w_o,    // head_num * head_dim * embedding_dim
        half *k_cache,// batch * head_num * ((seq_len - 1 (+1)) * head_dim)^T
        half *v_cache,// batch * head_num * (seq_len - 1 (+1)) * head_dim
        float sm_scale,
        half *global_batch_reduction, // batch
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

    // preallocate SRAM. (be careful with SRAM usage)
    __shared__ float block_reduce_float[BLOCK_SIZE / 32];
    __shared__ float tmp_mem[8];
    __shared__ __align__
    (16)
    half local_HEAD_DIM[HEAD_DIM];
    __shared__ __align__
    (16)
    half local_HEAD_DIM_prev[HEAD_DIM];
    __shared__ __align__
    (16)
    half local_Q[HEAD_DIM];
    __shared__ __align__
    (16)
    half local_FL_DEC_SPLIT[FL_DEC_SPLIT];
    __shared__ __align__
    (16)
    half block_reduce_half[BLOCK_SIZE / 32];
    __shared__ __align__
    (16)
    float local_slice_scale[SMALL_COOPERATE_BLOCK_NUM];

    // load Q [1 x HEAD_DIM] to share memory
    if (tid < HEAD_DIM / 8) {
        *(uint4 * )(&local_Q[tid * 8]) = *(uint4 * )(
                &q[batch_idx * HEAD_DIM * HEAD_NUM + head_idx * HEAD_DIM + tid * 8]);
    }
    __syncthreads();

    __align__(16)
    half kv_cache_reg[8], local_q_reg[8], local_s_reg[8], local_o_reg[8], local_prev_o_reg[8];
    __align__(16)
    float local_qk_value[(FL_DEC_SPLIT + BLOCK_SIZE - 1) / BLOCK_SIZE];
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
    for (int iter = 0; cooperate_idx < active_cooperator && iter < WORKLOAD / FL_DEC_SPLIT; iter++) {
        local_scale = 0; // chunk scale
        for (int j = tid; j < FL_DEC_SPLIT; j += BLOCK_SIZE) {
            int cache_idx = cooperate_idx * WORKLOAD + iter * FL_DEC_SPLIT + j;
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
        // [NOTE]: pass simple check

        // v
        for (int j = tid; j < HEAD_DIM; j += BLOCK_SIZE) {
            half sum_reg = 0;
            for (int di = 0; di < FL_DEC_SPLIT / 8; ++di) {
                int cache_idx = cooperate_idx * WORKLOAD + iter * FL_DEC_SPLIT + 8 * di;
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
        // [NOTE]: pass simple check

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
}


int main(int argc, char **argv) {
    // shared memory size per threadBlock
    size_t dynamicShMemSize = sizeof(float) * BLOCK_SIZE + sizeof(half) * (HEAD_DIM * 3 + EMBEDDING_DIM);
    cudaFuncSetAttribute(attn, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicShMemSize);

    float sm_scale = sqrt(HEAD_DIM);
    half *h_q, *d_q;
    half *h_k_cache, *d_k_cache;
    half *h_v_cache, *d_v_cache;
    half *h_w_o, *d_w_o;

    h_q = new half[BATCH_SIZE * HEAD_NUM * HEAD_DIM];
    h_w_o = new half[HEAD_NUM * HEAD_DIM * EMBEDDING_DIM];
    h_k_cache = new half[BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM];
    h_v_cache = new half[BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM];

    fill_matrix(h_q, BATCH_SIZE * HEAD_NUM * HEAD_DIM);
    fill_matrix(h_w_o, HEAD_NUM * HEAD_DIM * EMBEDDING_DIM);
    fill_matrix(h_k_cache, BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM);
    fill_matrix(h_v_cache, BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM);

    cudaMalloc(reinterpret_cast<void **>(&d_q), sizeof(half) * BATCH_SIZE * HEAD_NUM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_w_o), sizeof(half) * HEAD_NUM * HEAD_DIM * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_k_cache), sizeof(half) * BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_v_cache), sizeof(half) * BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM);

    cudaMemcpy(reinterpret_cast<void *>(d_q), h_q, sizeof(half) * BATCH_SIZE * HEAD_NUM * HEAD_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_o), h_w_o, sizeof(half) * HEAD_NUM * HEAD_DIM * EMBEDDING_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_k_cache), h_k_cache,
               sizeof(half) * BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_v_cache), h_v_cache,
               sizeof(half) * BATCH_SIZE * HEAD_NUM * SEQ_LEN * HEAD_DIM, cudaMemcpyHostToDevice);

    half *h_output, *d_output;
    h_output = new half[BATCH_SIZE * EMBEDDING_DIM];
    cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(half) * BATCH_SIZE * EMBEDDING_DIM);

    half h_global_batch[BATCH_SIZE] = {__float2half(0.0)};
    half h_global_reduce_attn[BATCH_SIZE * HEAD_NUM * (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM)];
    float h_global_reduce_attn_scale[BATCH_SIZE * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM];
    half *global_reduce_batch;
    half *global_reduce_attn;
    float *global_reduce_attn_scale;
    cudaMalloc(reinterpret_cast<void **>(&global_reduce_batch), sizeof(half) * BATCH_SIZE);
    cudaMalloc(reinterpret_cast<void **>(&global_reduce_attn), sizeof(half) * BATCH_SIZE * HEAD_NUM *
                                                               (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM));
    cudaMalloc(reinterpret_cast<void **>(&global_reduce_attn_scale),
               sizeof(float) * BATCH_SIZE * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM);
    cudaMemcpy(reinterpret_cast<void *>(global_reduce_batch), h_global_batch,
               sizeof(half) * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(global_reduce_attn), h_global_reduce_attn,
               sizeof(half) * BATCH_SIZE * HEAD_NUM * (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM),
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(global_reduce_attn_scale), h_global_reduce_attn_scale,
               sizeof(float) * BATCH_SIZE * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM, cudaMemcpyHostToDevice);

    void *kernelArgs[] = {
            &d_output, &d_q,
            &d_w_o,
            &d_k_cache, &d_v_cache,
            &sm_scale,
            &global_reduce_batch,
            &global_reduce_attn,
            &global_reduce_attn_scale
    };

    /**
     * ! Without cluster, we have to do heavy global sync in grid scope.
     *      So we try to do less intro-block sync in the kernel.
     */
    dim3 grid(BATCH_SIZE, HEAD_NUM, SMALL_COOPERATE_BLOCK_NUM);
    dim3 block(BLOCK_SIZE);

    cudaLaunchCooperativeKernel((void *) attn, grid, block, kernelArgs, dynamicShMemSize);


    cudaMemcpy(h_global_reduce_attn, reinterpret_cast<void *>(global_reduce_attn),
               sizeof(half) * BATCH_SIZE * HEAD_NUM * (SMALL_COOPERATE_BLOCK_NUM * HEAD_DIM),
               cudaMemcpyDeviceToHost);
    for (int bs = 0; bs < BATCH_SIZE; ++bs) {
        for (int h = 0; h < HEAD_NUM; ++h) {
            for (int i = 0; i < HEAD_DIM; ++i) {
                std::cout << __half2float(h_global_reduce_attn[bs * HEAD_NUM * HEAD_DIM + h * HEAD_DIM + i]) << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "\n";

    cudaMemcpy(h_global_reduce_attn_scale, reinterpret_cast<void *>(global_reduce_attn_scale),
               sizeof(float) * BATCH_SIZE * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM, cudaMemcpyDeviceToHost);
    for (int bs = 0; bs < BATCH_SIZE; ++bs) {
        for (int h = 0; h < HEAD_NUM; ++h) {
            for (int i = 0; i < SMALL_COOPERATE_BLOCK_NUM; ++i) {
                std::cout << h_global_reduce_attn_scale[bs * HEAD_NUM * SMALL_COOPERATE_BLOCK_NUM +
                                                        h * SMALL_COOPERATE_BLOCK_NUM + i] << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}