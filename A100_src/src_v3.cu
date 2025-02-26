#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>
#include <math.h>
#include <fstream>

namespace cg = cooperative_groups;

// nvcc -arch=sm_80 -std=c++17 src.cu -o test -Xptxas=-v -Xptxas=-warn-lmem-usage 

std::mt19937 rng(42);
std::normal_distribution<float> norm_dist(0.0, 0.1);

template<typename T>
void fill_matrix_from_file(T *mat, int sz) {
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

template<typename T>
void fill_matrix_from_file(T *mat, int sz, const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }
    
    for (int i = 0; i < sz; i++) {
        float value;
        if (!(file >> value)) { // 读取失败
            std::cerr << "Error: Not enough data in file" << std::endl;
            break;
        }
        
        if constexpr (std::is_same<T, __half>::value) {
            mat[i] = __float2half(value); // 转换为 half
        } else {
            mat[i] = value; // 直接存储 float / double
        }
    }
    std::cout << filename << " loaded!\n";
}

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

#define DEBUG 1
#define USE_GEMV_T_V1 1

/**
 * Decode config
 */
#define EMBEDDING_DIM 4096  // token embedding dimension
#define HEAD_NUM 32      // attn head number
#define HEAD_DIM (EMBEDDING_DIM/HEAD_NUM)   // attn head dimension
#define FFN_HIDDEN 11008     // ffn hidden dimension

#define SEQ_LEN 4096        // sequence length

/**
 * Kernel config
 */
#define WARP_SIZE 32
#define WARP_NUMBER 4 // 4 8 16
#define BLOCK_SIZE (WARP_SIZE*WARP_NUMBER)

#define FAKE_CLUSTER_NUM HEAD_NUM // number of blocks working on a request (>HEAD_NUM)
#define FAKE_CLUSTER_SIZE 4

/**
 * Other config
 */
#define EMBEDDING_TILE_LENGTH (EMBEDDING_DIM / FAKE_CLUSTER_SIZE)
#define EMBEDDING_TILE_IN_THREAD (EMBEDDING_DIM / FAKE_CLUSTER_SIZE / BLOCK_SIZE)

#define GEMV_VECTOR_SIZE EMBEDDING_TILE_LENGTH
#define GEMV_CHUNK_X HEAD_DIM
#define GEMV_CHUNK_Y 64

#define ATTN_TILE_LENGTH_IN_BLOCK (SEQ_LEN / FAKE_CLUSTER_SIZE) // flash decoding split
#define ATTN_ITER_LENGTH 1024 // flash attention split

#define BLOCK_SHARED_HALF MAX(2*GEMV_CHUNK_X*GEMV_CHUNK_Y + HEAD_DIM * 2 + ATTN_ITER_LENGTH, FAKE_CLUSTER_SIZE*HEAD_DIM + 8)
#define GLOBAL_REDUCE_HALF (3*FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM)

__device__ void rms_norm(
    half* embedding_reg, // local
    half *w_rms, // global
    float* block_reduce_float, // shared
    float* global_float_reduction // global
) {
    const float eps = 1e-5;
    cg::grid_group grid = cg::this_grid();
    const uint32_t block_idx_in_cluster = blockIdx.y;
    cg::thread_block block = cg::this_thread_block();
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; // 32 per warp
    const uint32_t warp_id = tid / WARP_SIZE;

    float local_tmp_float = 0.0f;
    #pragma unroll
    for (int di = 0; di < EMBEDDING_TILE_IN_THREAD; di++) {
        local_tmp_float += __half2float(embedding_reg[di]) * __half2float(embedding_reg[di]);
    }
    // block reduce
    #pragma unroll
    for (int stride = WARP_SIZE/2; stride > 0; stride >>= 1) {
        local_tmp_float += __shfl_down_sync(0xffffffff, local_tmp_float, stride);
    }
    if (lane_id == 0) {
        block_reduce_float[warp_id] = local_tmp_float;
    }
    __syncthreads();
    if (warp_id == 0) {
        if (lane_id < WARP_NUMBER) {
            local_tmp_float = block_reduce_float[lane_id];
        }
        #pragma unroll
        for (int stride = WARP_NUMBER/2; stride > 0; stride >>= 1) {
            local_tmp_float += __shfl_down_sync(0xffffffff, local_tmp_float, stride);
        }
        if (lane_id == 0){
            global_float_reduction[FAKE_CLUSTER_NUM * FAKE_CLUSTER_SIZE + block_idx_in_cluster] = local_tmp_float;
        }
    }
    // global reduce
    grid.sync();
    if (warp_id == 0) {
        if (lane_id < FAKE_CLUSTER_SIZE) {
            local_tmp_float = global_float_reduction[FAKE_CLUSTER_NUM * FAKE_CLUSTER_SIZE + lane_id];
        }
        #pragma unroll
        for (int stride = FAKE_CLUSTER_SIZE/2; stride > 0; stride >>= 1) {
            local_tmp_float += __shfl_down_sync(0xffffffff, local_tmp_float, stride);
        }
        if (lane_id == 0){
            block_reduce_float[0] = local_tmp_float;
        }
    }
    __syncthreads();
    local_tmp_float =  __float2half(1.f / (std::sqrt(block_reduce_float[0] / float(EMBEDDING_DIM))) + eps);
    __align__(16) half weight_reg[EMBEDDING_TILE_IN_THREAD];
    #pragma unroll
    for (int di = 0; di < EMBEDDING_TILE_IN_THREAD / 8; di++) {
        *(uint4 * )(&weight_reg[8 * di]) = *(uint4 * )(&w_rms[block_idx_in_cluster * EMBEDDING_TILE_LENGTH + tid * EMBEDDING_TILE_IN_THREAD + 8 * di]);
    }
    #pragma unroll
    for (int di = 0; di < EMBEDDING_TILE_IN_THREAD; di++) {
        embedding_reg[di] = __hmul(__hmul(embedding_reg[di], local_tmp_float), weight_reg[di]);
    }
}

/**
 * every thread load 8 half (16 bytes) per iteration
 */
__device__ void load_half_matrix_to_shared_mem(
    half *shared_addr, 
    half *matrix_starting_addr,
    uint32_t matrix_chunk_width, uint32_t matrix_chunk_height,
    uint32_t width_stride,
    uint32_t row_idx_offset,
    uint32_t column_idx_offset
) {
    cg::thread_block block = cg::this_thread_block();
    const uint32_t tid = block.thread_rank();

    uint32_t thread_per_row = (matrix_chunk_width >> 3);
    if (DEBUG && thread_per_row > WARP_SIZE) {
        printf("matrix_chunk_width = %d is not supported!", matrix_chunk_width);
    }
    uint32_t column_increment = BLOCK_SIZE / thread_per_row;

    uint32_t smem_ptr;

    uint32_t column_id = ((tid % thread_per_row) << 3);
    for(int iter = tid / thread_per_row; iter < matrix_chunk_height; iter += column_increment) {
        asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(&shared_addr[column_id + matrix_chunk_width * iter]));
        // 8 half
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr), // register
                        "l"(&matrix_starting_addr[column_id + column_idx_offset + width_stride * (row_idx_offset + iter)]), // long
                        "n"(16));
    } 
}

__device__ void gemv_tile(
    half *vector_starting_addr, // in shared memory
    half *matrix_chunk_starting_addr, // in shared memory
    float &sum,
    uint32_t buffer_id,
    uint32_t vector_offset,
    uint32_t matrix_chunk_width, uint32_t matrix_chunk_height
) {
    cg::thread_block block = cg::this_thread_block();
    const uint32_t tid = block.thread_rank();

    for(int inner_iter = 0; inner_iter < (matrix_chunk_height >> 3); ++inner_iter) {
        __align__(32) half local_vector_tile[8];
        *(uint4 * )(&local_vector_tile[0]) = *(uint4 * )(&vector_starting_addr[vector_offset + (inner_iter << 3)]);
        if(tid < matrix_chunk_width) {
            for(int ii = 0; ii < 8; ++ii) {
                sum += __half2float(__hmul(
                    local_vector_tile[ii],
                    matrix_chunk_starting_addr[
                        buffer_id * matrix_chunk_width * matrix_chunk_height + 
                        ((inner_iter<<3)+ii) * matrix_chunk_width + tid
                    ]
                ));
            }
        }
    }
}

__device__ void gemv_T_max_tile_v1(
    __half2 *local_q_tile, // in local
    half *matrix_chunk_starting_addr, // in shared memory
    half *output_starting_addr, // in shared memory
    float &maximum, float scale_factor,
    uint32_t buffer_id,
    uint32_t output_offset,
    uint32_t matrix_chunk_width, uint32_t matrix_chunk_height
) {
    cg::thread_block block = cg::this_thread_block();
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE;
    const uint32_t warp_id = tid / WARP_SIZE;
    const __half2* half2_matrix_ptr = reinterpret_cast<const __half2*>(&matrix_chunk_starting_addr[
        buffer_id * matrix_chunk_width * matrix_chunk_height
    ]);
    
    for(int i = 0; i < matrix_chunk_height / WARP_NUMBER; ++i) {
        float sum = 0.0f;
        #pragma unroll
        for(int inner_iter = 0; inner_iter < matrix_chunk_width / WARP_SIZE / 2; ++inner_iter) {
            __half2 value = __hmul2(
                local_q_tile[inner_iter], 
                half2_matrix_ptr[
                    (warp_id * matrix_chunk_height / WARP_NUMBER + i) * matrix_chunk_width / 2 + 
                    lane_id * matrix_chunk_width / WARP_SIZE / 2 + inner_iter
                ]
            );
            sum += __half2float(__hadd(__low2half(value), __high2half(value)));
        }
        // warp reduce
        #pragma unroll
        for (int stride = WARP_SIZE/2; stride > 0; stride >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, stride);
        }
        if(lane_id == 0){
            sum /= scale_factor;
            maximum = max(maximum, sum);
            output_starting_addr[
                output_offset + warp_id * matrix_chunk_height / WARP_NUMBER + i
            ] = __float2half(sum);
        }
    }
}

__global__ void single_decode(
    half *output, // embedding_dim
    half *input,  // embedding_dim
    half *w_rms_input, // embedding_dim
    half *w_rms_attn, // embedding_dim
    float* rope_cos, // head_dim
    float* rope_sin, // head_dim
    half *w_q,    // embedding_dim * head_num * head_dim = embedding_dim * embedding_dim
    half *w_k,    // embedding_dim * head_num * head_dim = embedding_dim * embedding_dim
    half *w_v,    // embedding_dim * head_num * head_dim = embedding_dim * embedding_dim
    half *w_o,    // head_num * head_dim * embedding_dim = embedding_dim * embedding_dim
    half *k_cache, // (seq_len - 1 (+1)) * head_num * head_dim
    half *v_cache, // (seq_len - 1 (+1)) * head_num * head_dim
    half *ffn_gate, // embedding_dim * ffn_hidden
    half *ffn_down, // ffn_hidden * embedding_dim
    half *ffn_up, // embedding_dim * ffn_hidden
    half *global_head_dim, // head_num * head_dim
    half *global_reduction
) {
    float* global_float_reduction = reinterpret_cast<float*>(global_reduction);

    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    const uint32_t head_idx = blockIdx.x;
    const uint32_t block_idx_in_cluster = blockIdx.y;
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE;
    const uint32_t warp_id = tid / WARP_SIZE;

    __shared__ __align__(16) float block_reduce_float[WARP_NUMBER];
    __shared__ __align__(16) half block_shared_half[BLOCK_SHARED_HALF];
    __shared__ __align__(16) half shared_vector[GEMV_VECTOR_SIZE];
    
    // #------------------------------------------------------------------------
    // # RMS Norm
    // #------------------------------------------------------------------------
    __align__(16) half embedding_reg[EMBEDDING_TILE_IN_THREAD];
    #pragma unroll
    for (int di = 0; di < EMBEDDING_TILE_IN_THREAD / 8; di++) {
        *(uint4 * )(&embedding_reg[8 * di]) = *(uint4 * )(&input[block_idx_in_cluster * EMBEDDING_TILE_LENGTH + tid * EMBEDDING_TILE_IN_THREAD + 8 * di]);
    }
    rms_norm(embedding_reg, w_rms_input, block_reduce_float, global_float_reduction);
    // store to shared memory
    #pragma unroll
    for (int di = 0; di < EMBEDDING_TILE_IN_THREAD / 8; di++) {
        *(uint4 * )(&shared_vector[tid * EMBEDDING_TILE_IN_THREAD + 8 * di]) = *(uint4 * )(&embedding_reg[8 * di]);
    }
    __syncthreads();
    // #------------------------------------------------------------------------

    // #------------------------------------------------------------------------
    // # W_q | W_k | W_v x
    // # ~~~ block_shared_half ~~~
    // # |  weight buffer  |  rope tmp (head dim)  |
    // # ~~~~~~~~~~~~~~~~~~~~~~~~~
    // #------------------------------------------------------------------------
    half* shared_rope_ptr = &block_shared_half[2*GEMV_CHUNK_X*GEMV_CHUNK_Y];
    int iter = 0;
    float sum = 0.0f, peer = 0.0f;
    float cos, sin;
    // GEMV with double buffer
    // Q header
    load_half_matrix_to_shared_mem(
        block_shared_half, w_q, GEMV_CHUNK_X, GEMV_CHUNK_Y,
        HEAD_DIM*HEAD_NUM, block_idx_in_cluster * EMBEDDING_TILE_LENGTH, HEAD_DIM * head_idx
    );
    asm volatile("cp.async.commit_group;\n" ::);
    // Q loop body
    for (; iter < (EMBEDDING_TILE_LENGTH / GEMV_CHUNK_Y) - 1; ++iter) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        load_half_matrix_to_shared_mem(
            &block_shared_half[(iter + 1)%2 * GEMV_CHUNK_X * GEMV_CHUNK_Y], 
            w_q, GEMV_CHUNK_X, GEMV_CHUNK_Y,
            HEAD_DIM*HEAD_NUM, block_idx_in_cluster * EMBEDDING_TILE_LENGTH + (iter + 1) * GEMV_CHUNK_Y, HEAD_DIM * head_idx
        );
        asm volatile("cp.async.commit_group;\n" ::);

        // calculate
        gemv_tile(
            shared_vector, block_shared_half, sum, iter%2, iter * GEMV_CHUNK_Y, GEMV_CHUNK_X, GEMV_CHUNK_Y
        );
    }
    // Q tile
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();
    // K header
    load_half_matrix_to_shared_mem(
        block_shared_half, w_k, GEMV_CHUNK_X, GEMV_CHUNK_Y,
        HEAD_DIM*HEAD_NUM, block_idx_in_cluster * EMBEDDING_TILE_LENGTH, HEAD_DIM * head_idx
    );
    asm volatile("cp.async.commit_group;\n" ::);
    gemv_tile(
        shared_vector, block_shared_half, sum, iter%2, iter * GEMV_CHUNK_Y, GEMV_CHUNK_X, GEMV_CHUNK_Y
    );

    if(tid < HEAD_DIM) {
        shared_rope_ptr[tid] = __float2half(sum);
    }
    __syncthreads();
    if(tid < HEAD_DIM) {
        peer = __half2float(shared_rope_ptr[tid < (HEAD_DIM / 2) ? tid + (HEAD_DIM / 2): tid - (HEAD_DIM / 2)]);
        cos = rope_cos[tid];
        sin = rope_sin[tid];
        global_reduction[head_idx * FAKE_CLUSTER_SIZE * HEAD_DIM + block_idx_in_cluster * HEAD_DIM + tid] = tid < (HEAD_DIM / 2) ?
            __float2half(cos * sum - sin * peer) : __float2half(cos * sum + sin * peer);
    }
    // K loop body
    for (iter = 0, sum = 0.0f; iter < (EMBEDDING_TILE_LENGTH / GEMV_CHUNK_Y) - 1; ++iter) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        load_half_matrix_to_shared_mem(
            &block_shared_half[(iter + 1)%2 * GEMV_CHUNK_X * GEMV_CHUNK_Y], 
            w_k, GEMV_CHUNK_X, GEMV_CHUNK_Y,
            HEAD_DIM*HEAD_NUM, block_idx_in_cluster * EMBEDDING_TILE_LENGTH + (iter + 1) * GEMV_CHUNK_Y, HEAD_DIM * head_idx
        );
        asm volatile("cp.async.commit_group;\n" ::);

        // calculate
        gemv_tile(
            shared_vector, block_shared_half, sum, iter%2, iter * GEMV_CHUNK_Y, GEMV_CHUNK_X, GEMV_CHUNK_Y
        );
    }
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();
    // V header
    load_half_matrix_to_shared_mem(
        block_shared_half, w_v, GEMV_CHUNK_X, GEMV_CHUNK_Y,
        HEAD_DIM*HEAD_NUM, block_idx_in_cluster * EMBEDDING_TILE_LENGTH, HEAD_DIM * head_idx
    );
    asm volatile("cp.async.commit_group;\n" ::);
    gemv_tile(
        shared_vector, block_shared_half, sum, iter%2, iter * GEMV_CHUNK_Y, GEMV_CHUNK_X, GEMV_CHUNK_Y
    );
    if(tid < HEAD_DIM) {
        shared_rope_ptr[tid] = __float2half(sum);
    }
    __syncthreads();
    if(tid < HEAD_DIM) {
        peer = __half2float(shared_rope_ptr[tid < (HEAD_DIM / 2) ? tid + (HEAD_DIM / 2): tid - (HEAD_DIM / 2)]);
        global_reduction[FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM + head_idx * FAKE_CLUSTER_SIZE * HEAD_DIM + block_idx_in_cluster * HEAD_DIM + tid] = tid < (HEAD_DIM / 2) ?
            __float2half(cos * sum - sin * peer) : __float2half(cos * sum + sin * peer);
    }
    // V loop body
    for (iter = 0, sum = 0.0f; iter < (EMBEDDING_TILE_LENGTH / GEMV_CHUNK_Y) - 1; ++iter) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        load_half_matrix_to_shared_mem(
            &block_shared_half[(iter + 1)%2 * GEMV_CHUNK_X * GEMV_CHUNK_Y], 
            w_v, GEMV_CHUNK_X, GEMV_CHUNK_Y,
            HEAD_DIM*HEAD_NUM, block_idx_in_cluster * EMBEDDING_TILE_LENGTH + (iter + 1) * GEMV_CHUNK_Y, HEAD_DIM * head_idx
        );
        asm volatile("cp.async.commit_group;\n" ::);

        // calculate
        gemv_tile(
            shared_vector, block_shared_half, sum, iter%2, iter * GEMV_CHUNK_Y, GEMV_CHUNK_X, GEMV_CHUNK_Y
        );
    }
    // V tile
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();
    gemv_tile(
        shared_vector, block_shared_half, sum, iter%2, iter * GEMV_CHUNK_Y, GEMV_CHUNK_X, GEMV_CHUNK_Y
    );
    if(tid < HEAD_DIM) {
        global_reduction[
            2*FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM + 
            head_idx * FAKE_CLUSTER_SIZE * HEAD_DIM + 
            block_idx_in_cluster * HEAD_DIM + tid
        ] = __float2half(sum);
    }
    // QKV global reduction (can be parallelized, but require global memory)
    grid.sync();
    // block_id 0: Q -> global_head_dim
    // block_id 1: K -> append to k cache
    // block_id 2: V -> append to v cache
    if(block_idx_in_cluster < 3) {
        if (warp_id < HEAD_DIM/(2*WARP_SIZE)) {
            const __half2* half2_ptr = reinterpret_cast<const __half2*>(&global_reduction[
                block_idx_in_cluster * FAKE_CLUSTER_NUM * FAKE_CLUSTER_SIZE * HEAD_DIM + 
                head_idx * FAKE_CLUSTER_SIZE * HEAD_DIM
            ]);
            __half2* half2_output_ptr;
            if (block_idx_in_cluster == 0) {
                half2_output_ptr = reinterpret_cast<__half2*>(&global_head_dim[head_idx * HEAD_DIM]);
            } else if (block_idx_in_cluster == 1) {
                half2_output_ptr = reinterpret_cast<__half2*>(&k_cache[(SEQ_LEN - 1) * HEAD_DIM * HEAD_NUM + head_idx * HEAD_DIM]);
            } else if (block_idx_in_cluster == 2) {
                half2_output_ptr = reinterpret_cast<__half2*>(&v_cache[(SEQ_LEN - 1) * HEAD_DIM * HEAD_NUM + head_idx * HEAD_DIM]);
            }
            __half2 val = half2_ptr[warp_id * WARP_SIZE + lane_id];
            #pragma unroll
            for (int ii = 1; ii < FAKE_CLUSTER_SIZE; ++ii) {
                val = __hadd2(val, half2_ptr[ii * (HEAD_DIM/2) + warp_id * WARP_SIZE + lane_id]); 
            }
            half2_output_ptr[warp_id * WARP_SIZE + lane_id] = val;
        }
    }
    grid.sync();
    // #------------------------------------------------------------------------

    // #------------------------------------------------------------------------
    // # Attention
    // # ~~~~~~~~~~~~~~~~~~~~~~~~
    // # |  weight buffer  |  Q (head dim)  |  attn score (head dim)  |  O (head dim)  |
    // #------------------------------------------------------------------------
    half* shared_q_ptr = &block_shared_half[2*GEMV_CHUNK_X*GEMV_CHUNK_Y];
    half* shared_attn_score_ptr = &block_shared_half[2*GEMV_CHUNK_X*GEMV_CHUNK_Y + HEAD_DIM];
    half* shared_o_ptr = &block_shared_half[2*GEMV_CHUNK_X*GEMV_CHUNK_Y + 2 * HEAD_DIM];
    // load Q to shared memory
    if(tid < HEAD_DIM / 2) {
        *(__half2 * )(&shared_q_ptr[tid * 2]) = *(__half2 * )(&global_head_dim[head_idx * HEAD_DIM + tid * 2]);
    }
    __half2 local_head_dim_tile[HEAD_DIM/WARP_SIZE/2]; // load q to every warp
    if(USE_GEMV_T_V1) {
        // every warp has full q
        __syncthreads();
        const __half2* half2_ptr = reinterpret_cast<const __half2*>(shared_q_ptr);
        #pragma unroll
        for(int i = 0; i < HEAD_DIM/WARP_SIZE/2; ++i) {
            local_head_dim_tile[i] = half2_ptr[lane_id * HEAD_DIM/WARP_SIZE/2 + i];
        }
    }
    float prev_l = 0.0f, current_l;
    float prev_m = -100.0f, current_m;
    for(int outer_iter = 0; outer_iter < ATTN_TILE_LENGTH_IN_BLOCK / ATTN_ITER_LENGTH; ++outer_iter) {
        float local_max_attn_score = -100.0f;
        // Q @ K^T header
        load_half_matrix_to_shared_mem(
            block_shared_half, k_cache, GEMV_CHUNK_X, GEMV_CHUNK_Y,
            HEAD_DIM * HEAD_NUM, 
            block_idx_in_cluster * ATTN_TILE_LENGTH_IN_BLOCK + outer_iter * ATTN_ITER_LENGTH, 
            HEAD_DIM * head_idx
        );
        asm volatile("cp.async.commit_group;\n" ::);
        // Q @ K^T loop body
        int inner_iter = 0;
        for (; inner_iter < (ATTN_ITER_LENGTH / GEMV_CHUNK_Y) - 1; ++inner_iter) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
            __syncthreads();
            load_half_matrix_to_shared_mem(
                &block_shared_half[(inner_iter + 1)%2 * GEMV_CHUNK_X * GEMV_CHUNK_Y], 
                k_cache, GEMV_CHUNK_X, GEMV_CHUNK_Y,
                HEAD_DIM * HEAD_NUM, 
                (inner_iter + 1) * GEMV_CHUNK_Y + block_idx_in_cluster * ATTN_TILE_LENGTH_IN_BLOCK + outer_iter * ATTN_ITER_LENGTH, 
                HEAD_DIM * head_idx
            );
            asm volatile("cp.async.commit_group;\n" ::);

            // calculate
            if (USE_GEMV_T_V1) {
                gemv_T_max_tile_v1(
                    local_head_dim_tile, block_shared_half, shared_attn_score_ptr, 
                    local_max_attn_score, sqrtf((float)HEAD_DIM), inner_iter%2,
                    inner_iter * GEMV_CHUNK_Y, GEMV_CHUNK_X, GEMV_CHUNK_Y
                );
            } else {
                printf("Other version to be implemented\n");
            }
        }
        // Q @ K^T tile
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        if (USE_GEMV_T_V1) {
            gemv_T_max_tile_v1(
                local_head_dim_tile, block_shared_half, shared_attn_score_ptr, 
                local_max_attn_score, sqrtf((float)HEAD_DIM), inner_iter%2,
                inner_iter * GEMV_CHUNK_Y, GEMV_CHUNK_X, GEMV_CHUNK_Y
            );
        } else {
            printf("Other version to be implemented\n");
        }
        if (lane_id == 0) {
            block_reduce_float[warp_id] = local_max_attn_score;
        }
        __syncthreads();
        // rowmax
        current_m = prev_m;
        __align__(16) float local_float[WARP_NUMBER];
        #pragma unroll
        for (int ii = 0; ii < WARP_NUMBER / 4; ++ii){
            *(uint4 * )(&local_float[ii * 4]) = *(uint4 * )(&block_reduce_float[ii * 4]);
        }
        #pragma unroll
        for (int ii = 0; ii < WARP_NUMBER; ++ii){
            current_m = max(current_m, local_float[ii]);
        }
        current_l = 0.0f;
        __align__(4) half local_attn_score[2];
        for(int ii = 0; ii < ATTN_TILE_LENGTH_IN_BLOCK/BLOCK_SIZE/2; ++ii) {
            *(__half2 * )(local_attn_score) = *(__half2 * )(&shared_attn_score_ptr[ii*BLOCK_SIZE*2 + tid *2]);
            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                float exp_ = exp(__half2float(local_attn_score[ki]) - current_m); // safe softmax
                local_attn_score[ki] = __float2half(exp_);
                current_l += exp_;
            }
            *(__half2 * )(&shared_attn_score_ptr[ii*BLOCK_SIZE*2 + tid *2]) = *(__half2 * )(local_attn_score);
        }
        #pragma unroll
        for (int stride = WARP_SIZE/2; stride > 0; stride >>= 1) {
            current_l += __shfl_down_sync(0xffffffff, current_l, stride);
        }
        if (lane_id == 0) {
            block_reduce_float[warp_id] = current_l;
        }
        __syncthreads();
        if (warp_id == 0) {
            if (lane_id < WARP_NUMBER) {
                current_l = block_reduce_float[lane_id];
            }
            #pragma unroll
            for (int stride = WARP_NUMBER/2; stride > 0; stride >>= 1) {
                current_l += __shfl_down_sync(0xffffffff, current_l, stride);
            }
            current_l += prev_l * exp(prev_m - current_m); // only tid==0 has correct current_l
        }
        // score * V header
        load_half_matrix_to_shared_mem(
            block_shared_half, v_cache, GEMV_CHUNK_X, GEMV_CHUNK_Y,
            HEAD_DIM*HEAD_NUM, 
            block_idx_in_cluster * ATTN_TILE_LENGTH_IN_BLOCK + outer_iter * ATTN_ITER_LENGTH, 
            HEAD_DIM * head_idx
        );
        asm volatile("cp.async.commit_group;\n" ::);
        // score * V loop body
        inner_iter = 0;
        float sum = 0.0f;
        for (; inner_iter < (ATTN_ITER_LENGTH / GEMV_CHUNK_Y) - 1; ++inner_iter) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
            __syncthreads();
            load_half_matrix_to_shared_mem(
                &block_shared_half[(inner_iter + 1)%2 * GEMV_CHUNK_X * GEMV_CHUNK_Y], 
                v_cache, GEMV_CHUNK_X, GEMV_CHUNK_Y,
                HEAD_DIM * HEAD_NUM, 
                (inner_iter + 1) * GEMV_CHUNK_Y + block_idx_in_cluster * ATTN_TILE_LENGTH_IN_BLOCK + outer_iter * ATTN_ITER_LENGTH, 
                HEAD_DIM * head_idx
            );
            asm volatile("cp.async.commit_group;\n" ::);

            // calculate
            gemv_tile(
                shared_attn_score_ptr, block_shared_half, sum, inner_iter%2, 
                inner_iter * GEMV_CHUNK_Y, GEMV_CHUNK_X, GEMV_CHUNK_Y
            );
        }
        // score * V tile
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        gemv_tile(
            shared_attn_score_ptr, block_shared_half, sum, inner_iter%2, 
            inner_iter * GEMV_CHUNK_Y, GEMV_CHUNK_X, GEMV_CHUNK_Y
        );
        // update attention output
        for (int ii = tid; ii < HEAD_DIM; ii += BLOCK_SIZE) { // tid < BLOCK_SIZE
            // shared_o_ptr[ii] = __float2half(sum / current_l);
            shared_o_ptr[ii] = __float2half(__half2float(shared_o_ptr[ii]) * exp(prev_m - current_m) + sum);
        }
        prev_m = current_m;
        prev_l = current_l;
    }
    // attention value reduce
    // # ~~~ global_reduction ~~~
    // # |  attn value (FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM)  |  partial max:l (FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*2)  |
    // # ~~~~~~~~~~~~~~~~~~~~~~~~
    half* global_attn_value = &global_reduction[head_idx*FAKE_CLUSTER_SIZE*HEAD_DIM + block_idx_in_cluster*HEAD_DIM];
    half* global_partial = &global_reduction[FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM+head_idx*FAKE_CLUSTER_SIZE*2];
    if (tid == 0) {
        global_partial[2*block_idx_in_cluster] = __float2half(current_m);
        global_partial[2*block_idx_in_cluster + 1] = __float2half(current_l);
    }
    // store output in shared memory to global
    if(warp_id == 0) {
        #pragma unroll
        for(int ii = 0; ii < HEAD_DIM/WARP_SIZE/2; ++ii) {
            *(__half2 * )(&global_attn_value[ii * WARP_SIZE * 2 + lane_id * 2]) = *(__half2 * )(&shared_o_ptr[ii * WARP_SIZE * 2 + lane_id * 2]);
        }
    }   
    grid.sync(); 
    // # ~~~ block_shared_half ~~~
    // # |  partial attn value (FAKE_CLUSTER_SIZE*HEAD_DIM)  |  max  |  l  |  scale factor(FAKE_CLUSTER_SIZE)  |
    // # ~~~~~~~~~~~~~~~~~~~~~~~~~
    if(block_idx_in_cluster == 0) {
        half* local_max = &block_shared_half[FAKE_CLUSTER_SIZE*HEAD_DIM];
        half* local_l = &block_shared_half[FAKE_CLUSTER_SIZE*HEAD_DIM + 1];
        half* local_scale_factors = &block_shared_half[FAKE_CLUSTER_SIZE*HEAD_DIM + 2];
        // async load partial attention into shared memory (warp_id > 0)
        if (warp_id > 0) { // thread_per_row = (HEAD_DIM >> 3) = 16;
            uint32_t smem_ptr;
            uint32_t column_id = ((lane_id % (HEAD_DIM >> 3)) << 3);
            for(int iter = warp_id - 1; iter < FAKE_CLUSTER_SIZE/(WARP_SIZE/(HEAD_DIM>>3)); iter += (WARP_NUMBER-1)) {
                uint32_t row_id = lane_id / (HEAD_DIM >> 3) + iter * (WARP_SIZE/(HEAD_DIM>>3));
                asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
                    : "=r"(smem_ptr)
                    : "l"(&block_shared_half[column_id + HEAD_DIM * row_id]));
                // 8 half
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr), // register
                                "l"(&global_attn_value[column_id + HEAD_DIM * row_id]), // long
                                "n"(16));
            }
            asm volatile("cp.async.commit_group;\n" ::);
        } else if (tid == 0) {
            __align__(16) half partial_values[FAKE_CLUSTER_SIZE*2];
            #pragma unroll
            for(int ii = 0; ii < FAKE_CLUSTER_SIZE/4; ++ii) {
                *(uint4 * )(&partial_values[ii * 8]) = *(uint4 * )(&global_partial[ii * 8]);
            }
            current_l = 0;
            #pragma unroll
            for(int ii = 0; ii < FAKE_CLUSTER_SIZE; ++ii) {
                current_m = max(current_m,__half2float(partial_values[ii*2]));
            }
            #pragma unroll
            for(int ii = 0; ii < FAKE_CLUSTER_SIZE; ++ii) {
                float scale_factor = exp(__half2float(partial_values[ii*2]) - current_m);
                current_l +=  scale_factor * __half2float(partial_values[ii*2+1]);
                local_scale_factors[ii] = __float2half(scale_factor);
            }
            local_max[0] = current_m;
            local_l[0] = current_l;
        }
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        // reduce to shared memory
        current_l = local_l[0];
        for (int ii = tid; ii < HEAD_DIM; ii+=BLOCK_SIZE) {
            float local_o = 0.0f;
            for (int inner = 0; inner < FAKE_CLUSTER_SIZE; ++inner) {
                local_o += __half2float(block_shared_half[HEAD_DIM*inner+ii])*__half2float(local_scale_factors[inner]);
            }
            block_shared_half[ii] = __float2half(local_o/current_l);
        }
        __syncthreads();
        // store to global
        if (warp_id < HEAD_DIM/(2*WARP_SIZE)) {
            *(__half2*)(&global_head_dim[
                head_idx * HEAD_DIM + (warp_id * WARP_SIZE + lane_id) * 2
            ]) = *(__half2*)(&block_shared_half[(warp_id * WARP_SIZE + lane_id) * 2]);
        }
    }
    grid.sync();
    // TODO
}

int main(int argc, char **argv) { 
    half *h_input, *d_input;
    half *h_k_cache, *d_k_cache;
    half *h_v_cache, *d_v_cache;
    half *h_w_q, *d_w_q;
    half *h_w_k, *d_w_k;
    half *h_w_v, *d_w_v;
    half *h_w_o, *d_w_o;
    half *h_ffn_gate, *d_ffn_gate;
    half *h_ffn_down, *d_ffn_down;
    half *h_ffn_up, *d_ffn_up;
    half *h_w_rms_input, *d_w_rms_input;
    half *h_w_rms_attn, *d_w_rms_attn;
    float *h_rope_cos, *d_rope_cos;
    float *h_rope_sin, *d_rope_sin;

    h_input = new half[EMBEDDING_DIM];
    h_w_q = new half[EMBEDDING_DIM * EMBEDDING_DIM];
    h_w_k = new half[EMBEDDING_DIM * EMBEDDING_DIM];
    h_w_v = new half[EMBEDDING_DIM * EMBEDDING_DIM];
    h_w_o = new half[EMBEDDING_DIM * EMBEDDING_DIM];
    h_k_cache = new half[HEAD_NUM * HEAD_DIM * SEQ_LEN ];
    h_v_cache = new half[HEAD_NUM * HEAD_DIM * SEQ_LEN ];
    h_ffn_gate = new half[EMBEDDING_DIM * FFN_HIDDEN];
    h_ffn_down = new half[FFN_HIDDEN * EMBEDDING_DIM];
    h_ffn_up = new half[EMBEDDING_DIM * FFN_HIDDEN];
    h_w_rms_input = new half[EMBEDDING_DIM];
    h_w_rms_attn = new half[EMBEDDING_DIM];
    h_rope_cos = new float[HEAD_DIM];
    h_rope_sin = new float[HEAD_DIM];

    // fill_matrix(h_input, EMBEDDING_DIM);
    // fill_matrix(h_w_q, EMBEDDING_DIM * EMBEDDING_DIM);
    // fill_matrix(h_w_k, EMBEDDING_DIM * EMBEDDING_DIM);
    // fill_matrix(h_w_v, EMBEDDING_DIM * EMBEDDING_DIM);
    // fill_matrix(h_w_o, EMBEDDING_DIM * EMBEDDING_DIM);
    // fill_matrix(h_k_cache, HEAD_NUM * HEAD_DIM * SEQ_LEN );
    // fill_matrix(h_v_cache, HEAD_NUM * HEAD_DIM * SEQ_LEN );
    // fill_matrix(h_ffn_gate, EMBEDDING_DIM * FFN_HIDDEN);
    // fill_matrix(h_ffn_down, FFN_HIDDEN * EMBEDDING_DIM);
    // fill_matrix(h_ffn_up, EMBEDDING_DIM * FFN_HIDDEN);
    // fill_matrix(h_w_rms_input, EMBEDDING_DIM);
    // fill_matrix(h_w_rms_attn, EMBEDDING_DIM);

    // fill rope sin/cos
    for (int i = 0; i < HEAD_DIM / 2; ++i) {
        float inv_freq = (SEQ_LEN / 1.0f) / (std::pow(float(1e4), float(2 * i) / float(HEAD_DIM)));
        h_rope_cos[i] = h_rope_cos[i + HEAD_DIM / 2] = std::cos(inv_freq);
        h_rope_sin[i] = h_rope_sin[i + HEAD_DIM / 2]  = std::sin(inv_freq);
    }

    cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(half) * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_w_q), sizeof(half) * EMBEDDING_DIM * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_w_k), sizeof(half) * EMBEDDING_DIM * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_w_v), sizeof(half) * EMBEDDING_DIM * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_w_o), sizeof(half) * EMBEDDING_DIM * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_k_cache), sizeof(half) * HEAD_NUM * HEAD_DIM * SEQ_LEN );
    cudaMalloc(reinterpret_cast<void **>(&d_v_cache), sizeof(half) * HEAD_NUM * HEAD_DIM * SEQ_LEN );
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_gate), sizeof(half) * EMBEDDING_DIM * FFN_HIDDEN);
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_down), sizeof(half) * FFN_HIDDEN * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_ffn_up), sizeof(half) * EMBEDDING_DIM * FFN_HIDDEN);
    cudaMalloc(reinterpret_cast<void **>(&d_w_rms_input), sizeof(half) * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_w_rms_attn), sizeof(half) * EMBEDDING_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_rope_cos), sizeof(float) * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void **>(&d_rope_sin), sizeof(float) * HEAD_DIM);

    cudaMemcpy(reinterpret_cast<void *>(d_input), h_input, sizeof(half) * EMBEDDING_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_q), h_w_q, sizeof(half) * EMBEDDING_DIM * EMBEDDING_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_k), h_w_k, sizeof(half) * EMBEDDING_DIM * EMBEDDING_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_v), h_w_v, sizeof(half) * EMBEDDING_DIM * EMBEDDING_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_o), h_w_o, sizeof(half) * EMBEDDING_DIM * EMBEDDING_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_k_cache), h_k_cache,
               sizeof(half) * HEAD_NUM * HEAD_DIM * SEQ_LEN , cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_v_cache), h_v_cache,
               sizeof(half) * HEAD_NUM * HEAD_DIM * SEQ_LEN , cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_ffn_gate), h_ffn_gate, sizeof(half) * EMBEDDING_DIM * FFN_HIDDEN,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_ffn_down), h_ffn_down, sizeof(half) * FFN_HIDDEN * EMBEDDING_DIM,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_ffn_up), h_ffn_up, sizeof(half) * EMBEDDING_DIM * FFN_HIDDEN,
               cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_rms_input), h_w_rms_input, sizeof(half) * EMBEDDING_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_w_rms_attn), h_w_rms_attn, sizeof(half) * EMBEDDING_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_rope_cos), h_rope_cos, sizeof(float) * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void *>(d_rope_sin), h_rope_sin, sizeof(float) * HEAD_DIM, cudaMemcpyHostToDevice);

    half *h_output, *d_output;
    h_output = new half[EMBEDDING_DIM];
    cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(half) * EMBEDDING_DIM);

    half *global_head_dim;
    cudaMalloc(reinterpret_cast<void **>(&global_head_dim), sizeof(half) * EMBEDDING_DIM);

    half *global_reduce;
    cudaMalloc(reinterpret_cast<void **>(&global_reduce), sizeof(half) * GLOBAL_REDUCE_HALF);

    void *kernelArgs[] = {
        &d_output, &d_input,
        &d_w_rms_input, &d_w_rms_attn,
        &d_rope_cos, &d_rope_sin,
        &d_w_q, &d_w_k, &d_w_v, &d_w_o,
        &d_k_cache, &d_v_cache,
        &d_ffn_gate, &d_ffn_down, &d_ffn_up,
        &global_head_dim,
        &global_reduce
    };

    dim3 grid(FAKE_CLUSTER_NUM, FAKE_CLUSTER_SIZE);
    dim3 block(WARP_NUMBER, WARP_SIZE);

    int warmup = 50;
    int test = 100;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);

    // * warm up
    for (int i = 0; i < warmup; i++) {
        cudaLaunchCooperativeKernel((void *) single_decode, grid, block, kernelArgs);
    }

    // * test kernel
    cudaEventRecord(st);
    for (int i = 0; i < test; i++) {
        cudaLaunchCooperativeKernel((void *) single_decode, grid, block, kernelArgs);
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);

    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << (ms / (1.0 * test)) << " ms" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void *>(d_output), sizeof(half) * EMBEDDING_DIM, cudaMemcpyDeviceToHost);

    return 0;
}