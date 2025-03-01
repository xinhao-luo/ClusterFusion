#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <torch/extension.h>

namespace cg = cooperative_groups;

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define DEBUG 1
#define MISALIGNED 0
#define USE_GEMV_T_V1 1

/**
 * Decode config
 */
#define HIDDEN_DIM 4096  // token embedding dimension
#define HEAD_NUM 32      // attn head number
#define HEAD_DIM (HIDDEN_DIM/HEAD_NUM)   // attn head dimension
#define FFN_DIM 11008     // ffn hidden dimension

#define SEQ_LEN 4096        // sequence length

/**
 * Kernel config
 */
#define WARP_SIZE 32
#define NUM_WARPS 4 // 4 8 16
#define BLOCK_SIZE (WARP_SIZE*NUM_WARPS)

#define FAKE_CLUSTER_NUM HEAD_NUM // number of blocks working on a request (>HEAD_NUM)
#define FAKE_CLUSTER_SIZE 4

/**
 * Other config
 */
#define EMBEDDING_TILE_LENGTH (HIDDEN_DIM / FAKE_CLUSTER_SIZE) // i.e. DIM_PER_BLOCK, rename it if you want
#define EMBEDDING_TILE_IN_THREAD (HIDDEN_DIM / FAKE_CLUSTER_SIZE / BLOCK_SIZE)

#define GEMV_VECTOR_SIZE EMBEDDING_TILE_LENGTH
#define FAKE_TMA_LOAD_ONCE 64
#define ATTN_PROJ_GEMV_CHUNK_X 64
#define ATTN_PROJ_GEMV_CHUNK_Y HEAD_DIM

#define KV_DIM_PER_BLOCK (SEQ_LEN / FAKE_CLUSTER_SIZE) // flash decoding split

#define BLOCK_SHARED_HALF MAX(2*ATTN_PROJ_GEMV_CHUNK_X*ATTN_PROJ_GEMV_CHUNK_Y + HEAD_DIM + EMBEDDING_TILE_LENGTH, MAX(2*FAKE_TMA_LOAD_ONCE*HEAD_DIM + HEAD_DIM * 2, FAKE_CLUSTER_SIZE*HEAD_DIM + 8))
#define GLOBAL_REDUCE_HALF MAX(2*FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM + FAKE_CLUSTER_NUM*HEAD_DIM, HEAD_NUM*HEAD_DIM+HEAD_NUM*HEAD_DIM*FAKE_CLUSTER_SIZE)

__device__ void rms_norm(
    half* embedding_reg, // local
    half *w_rms, // global
    uint32_t reduct_number, // `reduct_number` of partial sums are stored in `global_float_reduction`
    float* block_reduce_float, // shared
    float* global_float_reduction // global
) {
    const float eps = 1e-5;
    cg::grid_group grid = cg::this_grid();
    const uint32_t cluster_block_id = blockIdx.y;
    cg::thread_block block = cg::this_thread_block();
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; // 32 per warp
    const uint32_t warp_id = tid / WARP_SIZE;

    float local_tmp_float;
    // global reduce
    if (warp_id == 0) {
        if (lane_id < reduct_number) {
            local_tmp_float = global_float_reduction[lane_id];
        }
        #pragma unroll
        for (int stride = reduct_number/2; stride > 0; stride >>= 1) {
            local_tmp_float += __shfl_down_sync(0xffffffff, local_tmp_float, stride);
        }
        if (lane_id == 0){
            block_reduce_float[0] = local_tmp_float;
        }
    }
    block.sync();
    local_tmp_float =  1.f / (std::sqrt(block_reduce_float[0] / float(HIDDEN_DIM))) + eps;
    __align__(16) half weight_reg[EMBEDDING_TILE_IN_THREAD];
    #pragma unroll
    for (int di = 0; di < EMBEDDING_TILE_IN_THREAD / 8; di++) {
        *(uint4 * )(&weight_reg[8 * di]) = *(uint4 * )(&w_rms[cluster_block_id * EMBEDDING_TILE_LENGTH + tid * EMBEDDING_TILE_IN_THREAD + 8 * di]);
    }
    #pragma unroll
    for (int di = 0; di < EMBEDDING_TILE_IN_THREAD; di++) {
        embedding_reg[di] = __hmul(__hmul(embedding_reg[di], __float2half(local_tmp_float)), weight_reg[di]);
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

__device__ void x_Wt_tile(
    half *vector_starting_addr, // in shared memory
    half *matrix_chunk_starting_addr, // in shared memory
    float &sum,
    uint32_t buffer_id,
    uint32_t vector_offset,
    uint32_t matrix_chunk_width, uint32_t matrix_chunk_height
) {
    cg::thread_block block = cg::this_thread_block();
    const uint32_t tid = block.thread_rank();

    if(tid < matrix_chunk_height) {
        for(int inner_iter = 0; inner_iter < (matrix_chunk_width >> 3); ++inner_iter) {
            __align__(32) half local_vector_tile[8];
            __align__(32) half local_weight_tile[8];
            *(uint4 * )(&local_vector_tile[0]) = *(uint4 * )(&vector_starting_addr[vector_offset + (inner_iter << 3)]);
            *(uint4 * )(&local_weight_tile[0]) = *(uint4 * )(&matrix_chunk_starting_addr[
                buffer_id * matrix_chunk_width * matrix_chunk_height + tid * matrix_chunk_width + (inner_iter<<3)
            ]);
            for(int ii = 0; ii < 8; ++ii) {
                sum += __half2float(__hmul(local_vector_tile[ii], local_weight_tile[ii]));
            }
        }
    }
}

__device__ void x_W_tile(
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

__global__ void single_decode_kernel(
    half* output, // 1 * hidden_dim
    half* input,  // 1 * hidden_dim
    half* global_reduce,    // hidden_dim  
    half* w_rms_input,// hidden_dim
    half* w_rms_attn, // hidden_dim
    float* cos,       // head_dim
    float* sin,       // head_dim
    half *w_qkv,   // 3 * hidden_dim * hidden_dim
    half *kv_cache, // 2 * seqlen * head_num * head_dim
    half *w_o, // hidden_dim * hidden_dim
    half *ffn_gate_up, // 2 * hidden_dim * ffn_dim
    half *ffn_down // ffn_dim * hidden_dim
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    const uint32_t head_idx = blockIdx.x;
    const uint32_t cluster_block_id = blockIdx.y;
    const uint32_t tid = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE;
    const uint32_t warp_id = tid / WARP_SIZE;

    float eps = 1e-5;

    __shared__ __align__(16) half input_shmem[EMBEDDING_TILE_LENGTH];
    __shared__ __align__(16) half q_shmem[HEAD_DIM];
    __shared__ __align__(16) half kv_shmem[HEAD_DIM];
    __shared__ __align__(16) half attn_weight_shmem[KV_DIM_PER_BLOCK];

    __shared__ __align__(16) float block_reduce_float[NUM_WARPS];
    __shared__ __align__(16) half block_shared_half[BLOCK_SHARED_HALF];

    half* k_cache = kv_cache;
    half* v_cache = &kv_cache[SEQ_LEN*HEAD_DIM*HEAD_NUM];
    
    // Load input [1 x EMBEDDING_TILE_LENGTH] to shared memory
    #pragma unroll
    for (int i = tid * 8; i < EMBEDDING_TILE_LENGTH; i+=BLOCK_SIZE * 8) {
        *(uint4*)(&input_shmem[i]) = *(uint4*)(&input[cluster_block_id * EMBEDDING_TILE_LENGTH + i]);
    }
    block.sync();

    // #------------------------------------------------------------------------
    // # RMS Norm
    // #------------------------------------------------------------------------
    float* global_rms_reduction = reinterpret_cast<float*>(global_reduce);
    float local_sum = 0.0f;
    if (MISALIGNED) {
        __align__(16) half embedding_reg[EMBEDDING_TILE_IN_THREAD];
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_IN_THREAD / 8; di++) {
            *(uint4 * )(&embedding_reg[8 * di]) = *(uint4 * )(&input_shmem[tid * EMBEDDING_TILE_IN_THREAD + 8 * di]);
        }
        if(head_idx == 0) {
            #pragma unroll
            for (int di = 0; di < EMBEDDING_TILE_IN_THREAD; di++) {
                local_sum += __half2float(embedding_reg[di]) * __half2float(embedding_reg[di]);
            }
            #pragma unroll
            for (int mask = (WARP_SIZE>>1); mask > 0; mask >>= 1) {
                local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
            }
            if (lane_id == 0){
                block_reduce_float[warp_id] = local_sum;
            }
            block.sync(); 
            if (tid < NUM_WARPS) {
                local_sum = block_reduce_float[tid];
            }
            #pragma unroll
            for (int mask = (NUM_WARPS>>1); mask > 0; mask >>= 1) {
                local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
            }
            if (tid == 0) {
                global_rms_reduction[cluster_block_id] = local_sum;
            }
        }
        grid.sync();
        rms_norm(embedding_reg, w_rms_input, FAKE_CLUSTER_SIZE, block_reduce_float, global_rms_reduction);
        // store to shared memory
        #pragma unroll
        for (int di = 0; di < EMBEDDING_TILE_IN_THREAD / 8; di++) {
            *(uint4 * )(&input_shmem[tid * EMBEDDING_TILE_IN_THREAD + 8 * di]) = *(uint4 * )(&embedding_reg[8 * di]);
        }
        block.sync();
    } else {
        half __align__(16) reg_input_norm[2], reg_weight_norm[2];
        for (int d = tid * 2; d < EMBEDDING_TILE_LENGTH; d+=BLOCK_SIZE * 2) { 
            *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
            for (int di = 0; di < 2; di++)
                local_sum += __half2float(__hmul(reg_input_norm[di], reg_input_norm[di]));
        }
        #pragma unroll
        for (int mask = (WARP_SIZE>>1); mask > 0; mask >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
        }
        if (lane_id == 0){
            block_reduce_float[warp_id] = local_sum;
        }
        block.sync(); 
        if (tid < NUM_WARPS) {
            local_sum = block_reduce_float[tid];
        }
        #pragma unroll
        for (int mask = (NUM_WARPS>>1); mask > 0; mask >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
        }
        if (tid == 0) {
            global_rms_reduction[FAKE_CLUSTER_SIZE * head_idx + cluster_block_id] = local_sum;
        }
        grid.sync();
        // Reduce through Global
        if (warp_id == 0) {
            if (lane_id < FAKE_CLUSTER_SIZE) {
                local_sum = global_rms_reduction[FAKE_CLUSTER_SIZE * head_idx + lane_id];
            }
            #pragma unroll
            for (int stride = (FAKE_CLUSTER_SIZE>>1); stride > 0; stride >>= 1) {
                local_sum += __shfl_down_sync(0xffffffff, local_sum, stride);
            }
            if (lane_id == 0){
                block_reduce_float[0] = local_sum;
            }
        }
        block.sync();
        half rms_rcp =  __float2half(1.f / (std::sqrt(block_reduce_float[0] / float(HIDDEN_DIM))) + eps);
        for (int d = tid * 2; d < EMBEDDING_TILE_LENGTH; d+=BLOCK_SIZE * 2) { 
            *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
            *(half2*)(&reg_input_norm[0]) = __hmul2(*(half2*)(&reg_input_norm[0]), {rms_rcp, rms_rcp});
            *(half2*)(&reg_weight_norm[0]) = *(half2*)(&w_rms_input[EMBEDDING_TILE_LENGTH * cluster_block_id + d]);
            *(half2*)(&input_shmem[d]) = __hmul2(*(half2*)(&reg_input_norm[0]), *(half2*)(&reg_weight_norm[0]));
        }
        block.sync();
    }
    
    // #------------------------------------------------------------------------
    // # W_q | W_k x ([(head_num x head_dim) x hidden_dim])
    // #------------------------------------------------------------------------
    float sum = 0.0f;
    int iter = 0;
    // Q header
    load_half_matrix_to_shared_mem(
        block_shared_half, w_qkv, FAKE_TMA_LOAD_ONCE, HEAD_DIM,
        HIDDEN_DIM, HEAD_DIM * head_idx, cluster_block_id * EMBEDDING_TILE_LENGTH
    );
    asm volatile("cp.async.commit_group;\n" ::);
    // Q loop body
    for (; iter < (EMBEDDING_TILE_LENGTH / FAKE_TMA_LOAD_ONCE) - 1; ++iter) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        block.sync();
        load_half_matrix_to_shared_mem(
            &block_shared_half[(iter + 1)%2 * FAKE_TMA_LOAD_ONCE * HEAD_DIM], 
            w_qkv, FAKE_TMA_LOAD_ONCE, HEAD_DIM,
            HIDDEN_DIM, HEAD_DIM * head_idx, cluster_block_id * EMBEDDING_TILE_LENGTH + (iter + 1) * FAKE_TMA_LOAD_ONCE
        );
        asm volatile("cp.async.commit_group;\n" ::);
        x_Wt_tile(input_shmem, block_shared_half, sum, iter%2, iter * FAKE_TMA_LOAD_ONCE, FAKE_TMA_LOAD_ONCE, HEAD_DIM);
    }
    // Q tail
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    block.sync();
    x_Wt_tile(input_shmem, block_shared_half, sum, iter%2, iter * FAKE_TMA_LOAD_ONCE, FAKE_TMA_LOAD_ONCE, HEAD_DIM);
    // store partial Q in shmem
    if (tid < HEAD_DIM) {
        q_shmem[tid] = __float2half(sum);
    }
    
    sum = 0.0f;
    iter = 0;
    // K header
    load_half_matrix_to_shared_mem(
        block_shared_half, w_qkv, FAKE_TMA_LOAD_ONCE, HEAD_DIM,
        HIDDEN_DIM, HEAD_DIM * (HEAD_NUM + head_idx), cluster_block_id * EMBEDDING_TILE_LENGTH
    );
    asm volatile("cp.async.commit_group;\n" ::);
    // K loop body
    for (; iter < (EMBEDDING_TILE_LENGTH / FAKE_TMA_LOAD_ONCE) - 1; ++iter) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        block.sync();
        load_half_matrix_to_shared_mem(
            &block_shared_half[(iter + 1)%2 * FAKE_TMA_LOAD_ONCE * HEAD_DIM], 
            w_qkv, FAKE_TMA_LOAD_ONCE, HEAD_DIM,
            HIDDEN_DIM, HEAD_DIM * (HEAD_NUM + head_idx), cluster_block_id * EMBEDDING_TILE_LENGTH + (iter + 1) * FAKE_TMA_LOAD_ONCE
        );
        asm volatile("cp.async.commit_group;\n" ::);
        x_Wt_tile(input_shmem, block_shared_half, sum, iter%2, iter * FAKE_TMA_LOAD_ONCE, FAKE_TMA_LOAD_ONCE, HEAD_DIM);
    }
    // K tail
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    block.sync();
    x_Wt_tile(input_shmem, block_shared_half, sum, iter%2, iter * FAKE_TMA_LOAD_ONCE, FAKE_TMA_LOAD_ONCE, HEAD_DIM);
    // store partial K in shmem
    if (tid < HEAD_DIM) {
        kv_shmem[tid] = __float2half(sum);
    }
    block.sync();

    // rope
    if (MISALIGNED) {
        // TODO
    } else {
        half2 q_rope, q_rope_1;
        half2 k_rope, k_rope_1;
        float2 cos_reg, sin_reg;
        if (tid < HEAD_DIM / 2) {
            q_rope = *(half2*)(&q_shmem[tid * 2]);
            k_rope = *(half2*)(&kv_shmem[tid * 2]);
            if (tid * 2 < HEAD_DIM / 2) {
                q_rope_1 = *(half2*)(&q_shmem[HEAD_DIM / 2 + tid * 2]);
                k_rope_1 = *(half2*)(&kv_shmem[HEAD_DIM / 2 + tid * 2]);
                cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
                sin_reg = {-sin[HEAD_DIM / 2 + tid * 2], -sin[HEAD_DIM / 2 + tid * 2 + 1]};
            } else {
                q_rope_1 = *(half2*)(&q_shmem[tid * 2 - HEAD_DIM / 2]);
                k_rope_1 = *(half2*)(&kv_shmem[tid * 2 - HEAD_DIM / 2]);
                cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
                sin_reg = {sin[tid * 2 - HEAD_DIM / 2], sin[tid * 2 + 1 - HEAD_DIM / 2]};
            }
            *(half2*)(&q_shmem[tid * 2]) = __hadd2(__hmul2(q_rope, __float22half2_rn(cos_reg)), __hmul2(q_rope_1, __float22half2_rn(sin_reg)));
            *(half2*)(&kv_shmem[tid * 2]) = __hadd2(__hmul2(k_rope, __float22half2_rn(cos_reg)), __hmul2(k_rope_1, __float22half2_rn(sin_reg)));
        }
    }
    block.sync();
    
    // Q, K reduce (a little misaligned)
    // # ~~~ global_reduction ~~~
    // # |  q (FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM)  |  k (FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM)  |  reduced q (FAKE_CLUSTER_NUM*HEAD_DIM)
    // # ~~~~~~~~~~~~~~~~~~~~~~~~
    half* global_q_reduce = global_reduce;
    half* global_kv_reduce = &global_reduce[FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM];
    half* global_q = &global_reduce[2*FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM];
    if(warp_id < HEAD_DIM/(WARP_SIZE<<1)) {
        if (warp_id * WARP_SIZE + lane_id < (HEAD_DIM>>1)) {
            *(half2 *)(&global_q_reduce[
                head_idx*FAKE_CLUSTER_SIZE*HEAD_DIM+cluster_block_id*HEAD_DIM+2*(warp_id * WARP_SIZE + lane_id)
            ]) = *(half2 *)(&q_shmem[2*(warp_id * WARP_SIZE + lane_id)]);
        }
    } else if (warp_id < HEAD_DIM/(WARP_SIZE)) {
        if ((warp_id-(HEAD_DIM/(WARP_SIZE<<1))) * WARP_SIZE + lane_id < (HEAD_DIM>>1)) {
            *(half2 *)(&global_kv_reduce[
                head_idx*FAKE_CLUSTER_SIZE*HEAD_DIM+cluster_block_id*HEAD_DIM+2*((warp_id-HEAD_DIM/(WARP_SIZE<<1)) * WARP_SIZE + lane_id)
            ]) = *(half2 *)(&kv_shmem[2*((warp_id-HEAD_DIM/(WARP_SIZE<<1)) * WARP_SIZE + lane_id)]);
        }
    }
    grid.sync();
    if(cluster_block_id < 2) {
        if (warp_id < HEAD_DIM/(2*WARP_SIZE)) {
            const __half2* half2_ptr = reinterpret_cast<const __half2*>(&global_reduce[
                cluster_block_id * FAKE_CLUSTER_NUM * FAKE_CLUSTER_SIZE * HEAD_DIM + 
                head_idx * FAKE_CLUSTER_SIZE * HEAD_DIM
            ]);
            __half2* half2_output_ptr;
            if (cluster_block_id == 0) {
                half2_output_ptr = reinterpret_cast<__half2*>(&global_q[head_idx * HEAD_DIM]);
            } else if (cluster_block_id == 1) {
                half2_output_ptr = reinterpret_cast<__half2*>(&k_cache[(SEQ_LEN - 1) * HEAD_DIM * HEAD_NUM + head_idx * HEAD_DIM]);
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
    // # Q @ K^T
    // #------------------------------------------------------------------------
    float* global_softmax_reduction = reinterpret_cast<float*>(global_reduce); // FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*2
    float local_scale = 0.0f; // for softmax
    // load Q to shared memory
    if(tid < HEAD_DIM / 2) {
        *(__half2 * )(&q_shmem[tid * 2]) = *(__half2 * )(&global_q[head_idx * HEAD_DIM + tid * 2]);
    }
    // Q @ K^T header
    load_half_matrix_to_shared_mem(
        block_shared_half, k_cache, HEAD_DIM, FAKE_TMA_LOAD_ONCE,
        HIDDEN_DIM, cluster_block_id * KV_DIM_PER_BLOCK, HEAD_DIM * head_idx
    );
    asm volatile("cp.async.commit_group;\n" ::);
    // Q @ K^T loop body
    for (iter = 0; iter < (KV_DIM_PER_BLOCK / FAKE_TMA_LOAD_ONCE) - 1; ++iter) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        block.sync();
        load_half_matrix_to_shared_mem(
            &block_shared_half[(iter + 1)%2 * FAKE_TMA_LOAD_ONCE * HEAD_DIM], 
            k_cache, HEAD_DIM, FAKE_TMA_LOAD_ONCE,
            HIDDEN_DIM, cluster_block_id * KV_DIM_PER_BLOCK + (iter + 1) * FAKE_TMA_LOAD_ONCE, HEAD_DIM * head_idx
        );
        asm volatile("cp.async.commit_group;\n" ::);
        sum = 0.0f;
        x_Wt_tile(q_shmem, block_shared_half, sum, iter%2, 0, HEAD_DIM, FAKE_TMA_LOAD_ONCE);
        if(tid < FAKE_TMA_LOAD_ONCE) {
            float tmp = exp(sum/sqrt(1.0 * HEAD_DIM));
            local_scale += tmp;
            attn_weight_shmem[FAKE_TMA_LOAD_ONCE * iter + tid] = __float2half(tmp);
        }
    }
    // Q @ K^T tail
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    block.sync();
    sum = 0.0f;
    x_Wt_tile(q_shmem, block_shared_half, sum, iter%2, 0, HEAD_DIM, FAKE_TMA_LOAD_ONCE);
    if(tid < FAKE_TMA_LOAD_ONCE) {
        float tmp = exp(sum/sqrt(1.0 * HEAD_DIM));
        local_scale += tmp;
        attn_weight_shmem[FAKE_TMA_LOAD_ONCE * iter + tid] = __float2half(tmp);
    };
    #pragma unroll
    for (int mask = (WARP_SIZE>>1); mask > 0; mask >>= 1) {
        local_scale += __shfl_down_sync(0xffffffff, local_scale, mask);
    }
    if (lane_id == 0) {
        block_reduce_float[warp_id] = local_scale;
    }
    block.sync(); 
    if (tid < NUM_WARPS) {
        local_sum = block_reduce_float[tid];
    }
    #pragma unroll
    for (int mask = (NUM_WARPS>>1); mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }
    if (tid == 0) {
        global_softmax_reduction[FAKE_CLUSTER_SIZE * head_idx + cluster_block_id] = local_sum;
    }
    grid.sync();
    if (warp_id == 0) {
        if (lane_id < FAKE_CLUSTER_SIZE) {
            local_sum = global_softmax_reduction[FAKE_CLUSTER_SIZE * head_idx + lane_id];
        }
        #pragma unroll
        for (int stride = (FAKE_CLUSTER_SIZE>>1); stride > 0; stride >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, stride);
        }
        if (lane_id == 0){
            block_reduce_float[0] = local_sum;
        }
    } 
    block.sync();
    for (int i = tid; i < KV_DIM_PER_BLOCK; i+=BLOCK_SIZE) {
        attn_weight_shmem[i] = __float2half(__half2float(attn_weight_shmem[i]) / block_reduce_float[0]);
    }
    block.sync();

    // #------------------------------------------------------------------------
    // # W_v x ([(head_num x head_dim) x hidden_dim])
    // #------------------------------------------------------------------------
    sum = 0.0f;
    iter = 0;
    // V header
    load_half_matrix_to_shared_mem(
        block_shared_half, w_qkv, FAKE_TMA_LOAD_ONCE, HEAD_DIM,
        HIDDEN_DIM, HEAD_DIM * (HEAD_NUM*2 + head_idx), cluster_block_id * EMBEDDING_TILE_LENGTH
    );
    asm volatile("cp.async.commit_group;\n" ::);
    // V loop body
    for (; iter < (EMBEDDING_TILE_LENGTH / FAKE_TMA_LOAD_ONCE) - 1; ++iter) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        block.sync();
        load_half_matrix_to_shared_mem(
            &block_shared_half[(iter + 1)%2 * FAKE_TMA_LOAD_ONCE * HEAD_DIM], 
            w_qkv, FAKE_TMA_LOAD_ONCE, HEAD_DIM,
            HIDDEN_DIM, HEAD_DIM * (HEAD_NUM*2 + head_idx), cluster_block_id * EMBEDDING_TILE_LENGTH + (iter + 1) * FAKE_TMA_LOAD_ONCE
        );
        asm volatile("cp.async.commit_group;\n" ::);
        x_Wt_tile(input_shmem, block_shared_half, sum, iter%2, iter * FAKE_TMA_LOAD_ONCE, FAKE_TMA_LOAD_ONCE, HEAD_DIM);
    }
    // V tail
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    block.sync();
    x_Wt_tile(input_shmem, block_shared_half, sum, iter%2, iter * FAKE_TMA_LOAD_ONCE, FAKE_TMA_LOAD_ONCE, HEAD_DIM);
    // store partial V in shmem
    if (tid < HEAD_DIM) {
        kv_shmem[tid] = __float2half(sum);
    }
    block.sync();
    if (warp_id < HEAD_DIM/(WARP_SIZE<<1)) {
        if (warp_id * WARP_SIZE + lane_id < (HEAD_DIM>>1)) {
            *(half2 *)(&global_kv_reduce[
                head_idx*FAKE_CLUSTER_SIZE*HEAD_DIM+cluster_block_id*HEAD_DIM+2*(warp_id * WARP_SIZE + lane_id)
            ]) = *(half2 *)(&kv_shmem[2*(warp_id * WARP_SIZE + lane_id)]);
        }
    }
    grid.sync();
    // reduction (a little misaligned)
    if (cluster_block_id == 0 && warp_id < HEAD_DIM/(2*WARP_SIZE)) {
        const __half2* half2_ptr = reinterpret_cast<const __half2*>(&global_kv_reduce[head_idx * FAKE_CLUSTER_SIZE * HEAD_DIM]);
        __half2* half2_output_ptr = reinterpret_cast<__half2*>(&v_cache[(SEQ_LEN - 1) * HEAD_DIM * HEAD_NUM + head_idx * HEAD_DIM]);
        __half2 val = half2_ptr[warp_id * WARP_SIZE + lane_id];
        #pragma unroll
        for (int ii = 1; ii < FAKE_CLUSTER_SIZE; ++ii) {
            val = __hadd2(val, half2_ptr[ii * (HEAD_DIM/2) + warp_id * WARP_SIZE + lane_id]); 
        }
        half2_output_ptr[warp_id * WARP_SIZE + lane_id] = val;
    }
    grid.sync();

    // #------------------------------------------------------------------------
    // # attn weight @ V
    // #------------------------------------------------------------------------
    half* attention_output_shmem = q_shmem; // reuse 
    sum = 0.0f;
    load_half_matrix_to_shared_mem(
        block_shared_half, v_cache, HEAD_DIM, FAKE_TMA_LOAD_ONCE,
        HIDDEN_DIM, cluster_block_id * KV_DIM_PER_BLOCK, HEAD_DIM * head_idx
    );
    asm volatile("cp.async.commit_group;\n" ::);
    for (iter = 0; iter < (KV_DIM_PER_BLOCK / FAKE_TMA_LOAD_ONCE) - 1; ++iter){
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        block.sync();
        load_half_matrix_to_shared_mem(
            &block_shared_half[(iter + 1)%2 * FAKE_TMA_LOAD_ONCE * HEAD_DIM], 
            v_cache, HEAD_DIM, FAKE_TMA_LOAD_ONCE,
            HIDDEN_DIM, cluster_block_id * KV_DIM_PER_BLOCK + (iter + 1) * FAKE_TMA_LOAD_ONCE, HEAD_DIM * head_idx
        );
        asm volatile("cp.async.commit_group;\n" ::);
        x_W_tile(attn_weight_shmem, block_shared_half, sum, iter%2, iter * FAKE_TMA_LOAD_ONCE, HEAD_DIM, FAKE_TMA_LOAD_ONCE);
    }
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    block.sync();
    x_W_tile(attn_weight_shmem, block_shared_half, sum, iter%2, iter * FAKE_TMA_LOAD_ONCE, HEAD_DIM, FAKE_TMA_LOAD_ONCE);
    if (tid < HEAD_DIM) {
        attention_output_shmem[tid] = __float2half(sum);
    }
    block.sync();
    // # ~~~ global_reduction ~~~
    // # |  global_attn_output (FAKE_CLUSTER_NUM * HEAD_DIM)  |  global_attn_reduce (FAKE_CLUSTER_NUM*FAKE_CLUSTER_SIZE*HEAD_DIM)  |
    // # ~~~~~~~~~~~~~~~~~~~~~~~~
    half* global_attn_output = global_reduce; // FAKE_CLUSTER_NUM * HEAD_DIM
    half* global_attn_reduce = &(global_reduce[FAKE_CLUSTER_NUM * HEAD_DIM]);
    if (warp_id < HEAD_DIM/(WARP_SIZE<<1)) {
        if (warp_id * WARP_SIZE + lane_id < (HEAD_DIM>>1)) {
            *(half2 *)(&global_attn_reduce[
                head_idx*FAKE_CLUSTER_SIZE*HEAD_DIM+cluster_block_id*HEAD_DIM+2*(warp_id * WARP_SIZE + lane_id)
            ]) = *(half2 *)(&attention_output_shmem[2*(warp_id * WARP_SIZE + lane_id)]);
        }
    }
    grid.sync();
    if (cluster_block_id == 0 && warp_id < HEAD_DIM/(2*WARP_SIZE)) {
        const __half2* half2_ptr = reinterpret_cast<const __half2*>(&global_attn_reduce[head_idx * FAKE_CLUSTER_SIZE * HEAD_DIM]);
        __half2* half2_output_ptr = reinterpret_cast<__half2*>(&global_attn_output[head_idx * HEAD_DIM]);
        __half2 val = half2_ptr[warp_id * WARP_SIZE + lane_id];
        #pragma unroll
        for (int ii = 1; ii < FAKE_CLUSTER_SIZE; ++ii) {
            val = __hadd2(val, half2_ptr[ii * (HEAD_DIM/2) + warp_id * WARP_SIZE + lane_id]); 
        }
        half2_output_ptr[warp_id * WARP_SIZE + lane_id] = val;
    }
    grid.sync();

    // #------------------------------------------------------------------------
    // # Attention Projection
    // #------------------------------------------------------------------------
    half* attn_proj_output_shmem = &(block_shared_half[FAKE_TMA_LOAD_ONCE * HEAD_DIM * 2]); // EMBEDDING_TILE_LENGTH
    // load attention output to shared memory
    if(tid < HEAD_DIM / 2) {
        *(__half2 * )(&attention_output_shmem[tid * 2]) = *(__half2 * )(&global_attn_output[head_idx * HEAD_DIM + tid * 2]);
    }
    load_half_matrix_to_shared_mem(
        block_shared_half, w_o, HEAD_DIM, FAKE_TMA_LOAD_ONCE,
        HIDDEN_DIM, cluster_block_id * EMBEDDING_TILE_LENGTH , HEAD_DIM * head_idx
    );
    asm volatile("cp.async.commit_group;\n" ::);
    for (iter = 0; iter < (EMBEDDING_TILE_LENGTH / FAKE_TMA_LOAD_ONCE) - 1; ++iter) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        block.sync();
        load_half_matrix_to_shared_mem(
            &block_shared_half[(iter + 1)%2 * FAKE_TMA_LOAD_ONCE * HEAD_DIM], 
            w_o, HEAD_DIM, FAKE_TMA_LOAD_ONCE,
            HIDDEN_DIM, cluster_block_id * EMBEDDING_TILE_LENGTH + (iter + 1) * FAKE_TMA_LOAD_ONCE, HEAD_DIM * head_idx
        );
        asm volatile("cp.async.commit_group;\n" ::);
        sum = 0.0f;
        x_Wt_tile(attention_output_shmem, block_shared_half, sum, iter%2, 0, HEAD_DIM, FAKE_TMA_LOAD_ONCE);
        if(tid < FAKE_TMA_LOAD_ONCE) {
            attn_proj_output_shmem[FAKE_TMA_LOAD_ONCE * iter + tid] = __float2half(sum);
        }
    }
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    block.sync();
    sum = 0.0f;
    x_Wt_tile(attention_output_shmem, block_shared_half, sum, iter%2, 0, HEAD_DIM, FAKE_TMA_LOAD_ONCE);
    if(tid < FAKE_TMA_LOAD_ONCE) {
        attn_proj_output_shmem[FAKE_TMA_LOAD_ONCE * iter + tid] = __float2half(sum);
    }
    block.sync();
    // global reduce
    for(int ii = 0; ii < EMBEDDING_TILE_LENGTH/BLOCK_SIZE/2; ++ii) {
        *(__half2 * )(&global_reduce[
            head_idx*HIDDEN_DIM+cluster_block_id*EMBEDDING_TILE_LENGTH+
            ii*BLOCK_SIZE*2 + tid *2
        ])=*(__half2 * )(&attn_proj_output_shmem[ii*BLOCK_SIZE*2 + tid * 2]);
    }
    grid.sync();
    // (a little misaligned)
    // # ~~~ global_reduction ~~~
    // # |  projection output (HEAD_NUM*HIDDEN_DIM)  |  partial sum ((HIDDEN_DIM / FAKE_CLUSTER_SIZE / (64 * NUM_WARPS))*2) |
    // # ~~~~~~~~~~~~~~~~~~~~~~~~~
    float* global_partial_sums = reinterpret_cast<float*>(&global_reduce[HEAD_NUM*HIDDEN_DIM]);
    if(head_idx < HIDDEN_DIM / FAKE_CLUSTER_SIZE / (64 * NUM_WARPS)) { // 4
        __half2* input_shmem_half2_ptr = reinterpret_cast<__half2*>(input_shmem);
        __half2* global_projection_half2_ptr = reinterpret_cast<__half2*>(global_reduce);
        __half2 value = input_shmem_half2_ptr[(head_idx*64*NUM_WARPS)/2 + tid];
        for(int ii = 0; ii < HEAD_NUM; ++ii) {
            value = __hadd2(value, global_projection_half2_ptr[(ii * HIDDEN_DIM + cluster_block_id * EMBEDDING_TILE_LENGTH + head_idx * 64 * NUM_WARPS)/2 + tid]);
        }
        input_shmem_half2_ptr[tid] = value;
        float tmp1 = __half2float(__low2half(value)), tmp2 = __half2float(__high2half(value));
        float partial_sum = tmp1 * tmp1 + tmp2 *tmp2;
        // sum: warp reduce
        #pragma unroll
        for (int stride = WARP_SIZE/2; stride > 0; stride >>= 1) {
            partial_sum = partial_sum + __shfl_down_sync(0xffffffff, partial_sum, stride);
        }
        if(lane_id == 0) {
            block_reduce_float[warp_id] = partial_sum;
        }
        block.sync();
        // store to global
        if(tid < (BLOCK_SIZE*2)/8) {
            *(uint4 * )(&output[cluster_block_id * EMBEDDING_TILE_LENGTH + head_idx * 64 * NUM_WARPS + tid * 8]) = *(uint4 * )(&input_shmem[tid * 8]); 
        } else if(tid==(BLOCK_SIZE*2)/8) {
            partial_sum = 0.0f;
            #pragma unroll
            for(int ii = 0; ii < NUM_WARPS; ++ii) {
                partial_sum += block_reduce_float[ii];
            }
            global_partial_sums[head_idx * FAKE_CLUSTER_SIZE + cluster_block_id] = partial_sum;
        }
    }
    grid.sync(); 
    // now, `output` is written with attention output + residual
    __align__(16) half embedding_reg[EMBEDDING_TILE_IN_THREAD];
    #pragma unroll
    for (int di = 0; di < EMBEDDING_TILE_IN_THREAD / 8; di++) {
        *(uint4 * )(&embedding_reg[8 * di]) = *(uint4 * )(&output[cluster_block_id * EMBEDDING_TILE_LENGTH + tid * EMBEDDING_TILE_IN_THREAD + 8 * di]);
        *(uint4 * )(&input_shmem[tid * EMBEDDING_TILE_IN_THREAD + 8 * di]) = *(uint4 * )(&embedding_reg[8 * di]);
    }
    rms_norm(embedding_reg, w_rms_attn, (HIDDEN_DIM / (64 * NUM_WARPS)), block_reduce_float, global_partial_sums);
    // store to shared memory
    #pragma unroll
    for (int di = 0; di < EMBEDDING_TILE_IN_THREAD / 8; di++) {
        *(uint4 * )(&input_shmem[tid * EMBEDDING_TILE_IN_THREAD + 8 * di]) = *(uint4 * )(&embedding_reg[8 * di]);
    }
    block.sync();

    // #------------------------------------------------------------------------
    // # FFN
    // #------------------------------------------------------------------------
    // TODO
    
}

torch::Tensor single_decode_layer(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor kv_cache,
    torch::Tensor gate_up_proj_weight,
    torch::Tensor down_proj_weight,
    torch::Tensor rms_input_weight,
    torch::Tensor rms_attn_weight,
    torch::Tensor cos,
    torch::Tensor sin
) {
    // parameter
    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* weight_qkv_ptr = reinterpret_cast<half*>(weight_qkv.data_ptr<at::Half>());
    half* weight_o_ptr = reinterpret_cast<half*>(weight_o.data_ptr<at::Half>());
    half* kv_cache_ptr = reinterpret_cast<half*>(kv_cache.data_ptr<at::Half>());
    half* gate_up_proj_weight_ptr = reinterpret_cast<half*>(gate_up_proj_weight.data_ptr<at::Half>());
    half* down_proj_weight_ptr = reinterpret_cast<half*>(down_proj_weight.data_ptr<at::Half>());
    half* rms_input_weight_ptr = reinterpret_cast<half*>(rms_input_weight.data_ptr<at::Half>());
    half* rms_attn_weight_ptr = reinterpret_cast<half*>(rms_attn_weight.data_ptr<at::Half>());
    float* cos_ptr = reinterpret_cast<float*>(cos.data_ptr<float>());
    float* sin_ptr = reinterpret_cast<float*>(sin.data_ptr<float>());

    // global reduce
    half *reduce_workspace;
    cudaMalloc(reinterpret_cast<void**>(&reduce_workspace), sizeof(half) * 1 * HIDDEN_DIM);

    // output
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor output = torch::full({1, HIDDEN_DIM}, 0, options);
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    
    dim3 grid(FAKE_CLUSTER_NUM * FAKE_CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    single_decode_kernel<<<grid, block>>>(
        output_ptr,
        input_ptr,
        reduce_workspace,
        rms_input_weight_ptr,
        rms_attn_weight_ptr,
        cos_ptr,
        sin_ptr,
        weight_qkv_ptr,
        kv_cache_ptr,
        weight_o_ptr,
        gate_up_proj_weight_ptr,
        down_proj_weight_ptr
    );
    cudaDeviceSynchronize();

    return output;
}