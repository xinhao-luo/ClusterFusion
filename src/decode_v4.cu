#include "cuda_runtime.h"                                                                                                              
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <iostream>
#include <random>
#include <stdio.h>

// CUDA_VISIBLE_DEVICES=1 nvcc -arch=sm_90a -std=c++17 decode_v4.cu -o test && ./test 

#define HEAD_DIM 128    // attn head dimension
#define HEAD_NUM 32     // attn head number
#define FFN_HIDDEN 4096 // ffn hidden dimension
#define HIDDEN_DIM 4096 // token embedding dimension
#define SEQ_LEN 4096    // sequence length

#define NUM_WARPS 8
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE) // 256
#define CLUSTER_SIZE 4
#define NUM_PER_THREAD 8
#define NUM_ROW_PER_WARP (HEAD_DIM / NUM_WARPS) // 16
#define NUM_THREAD_PER_ROW (WARP_SIZE / NUM_ROW_PER_WARP) // 2
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW) // 16
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE) // 
#define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) // 
#define NUM_ROW_PER_WARP_2 (KV_DIM_PER_BLOCK / NUM_WARPS) // 128

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 5.0);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(0.01f);
        }   
    }   
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) single_decode(
    half* output, // 1 * hidden_dim
    half* input,  // 1 * hidden_dim
    half* w_q,    // hidden_dim * head_num * head_dim
    half* w_k,    // hidden_dim * head_num * head_dim
    half* w_v,    // hidden_dim * head_num * head_dim
    half* w_o,    // head_num * head_dim * hidden_dim
    half* k_cache,// head_num * seqlen * head_dim
    half* v_cache,// head_num * seqlen * head_dim
    half* ffn_gate,  // hidden_dim * ffn_hidden
    half* ffn_down,  // ffn_hidden * hidden_dim
    half* ffn_up,    // hidden_dim * ffn_hidden
    half* global,    // hidden_dim  
    half* w_rms_input,// hidden_dim
    half* w_rms_attn, // hidden_dim
    float* cos,       // head_dim
    float* sin        // head_dim
)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank() % HEAD_NUM;
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; // 32 per warp
    const uint32_t warp_id = tid / WARP_SIZE;

    // Load input [1 x HIDDEN_DIM / CLUSTR_SIZE] to shared memory
    __shared__ __align__(16) half input_shmem[DIM_PER_BLOCK];
    #pragma unroll
    for (int i = tid; i < DIM_PER_BLOCK; i+=BLOCK_SIZE) {
        input_shmem[i] = input[cluster_block_id * DIM_PER_BLOCK + i];
    }
    block.sync();

    // RMSNorm
    __shared__ float norm_reduction[16];
    float local_sum = 0;
    __shared__ float cluster_local_sum;
    half __align__(16) reg_input_norm[2], reg_weight_norm[2];
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
        for (int di = 0; di < 2; di++)
            local_sum += __half2float(reg_input_norm[di] * reg_input_norm[di]);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }
    if (lane_id == 0){
        norm_reduction[warp_id] = local_sum;
    }
    block.sync(); 
    if (tid < 16) 
        local_sum = norm_reduction[tid];
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    } 
    if (tid == 0)
        cluster_local_sum = local_sum;
    cluster.sync();
    // Reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            float* dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
            atomicAdd(dst_shmem, local_sum);
        }
        cluster.sync();
    }
    float eps = 1e-5;
    half rms_rcp = __float2half(1.f / (std::sqrt(cluster_local_sum / float(HIDDEN_DIM)) + eps));
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
        *(half2*)(&reg_input_norm[0]) = __hmul2(*(half2*)(&reg_input_norm[0]), {rms_rcp, rms_rcp});
        *(half2*)(&reg_weight_norm[0]) = *(half2*)(&w_rms_input[d]);
        *(half2*)(&input_shmem[d]) = __hmul2(*(half2*)(&reg_input_norm[0]), *(half2*)(&reg_weight_norm[0]));
    }
    block.sync();

    // For DSM reduce
    __shared__ __align__(16) half local_buffer[HEAD_DIM];

    // Compute hidden @ wq
    float tmp = 0.0;
    half __align__(16) reg_input[1][NUM_PER_THREAD];
    half __align__(16) reg_weight[1][NUM_PER_THREAD];
    __shared__ __align__(16) half local_q[HEAD_DIM];
    uint input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx = head_id * HEAD_DIM * HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP * HIDDEN_DIM + (lane_id / NUM_THREAD_PER_ROW) * HIDDEN_DIM + (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    for (int i = 0; i < DIM_PER_BLOCK; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0][0]) = *(uint4*)(&input_shmem[input_idx + i]);
        *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&w_q[weight_idx + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_q[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    cluster.sync();

    // Compute hidden @ wk
    __shared__ __align__(16) half local_kv[HEAD_DIM];
    tmp = 0.0;
    for (int i = 0; i < DIM_PER_BLOCK; i+=NUM_PER_ROW) { // 16
        *(uint4*)(&reg_input[0][0]) = *(uint4*)(&input_shmem[input_idx + i]);
        *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&w_k[weight_idx + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_kv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // Compute partial RoPE
    half2 q_rope, q_rope_1;
    half2 k_rope, k_rope_1;
    float2 cos_reg, sin_reg;
    if (tid < HEAD_DIM / 2) {
        q_rope = *(half2*)(&local_q[tid * 2]);
        k_rope = *(half2*)(&local_kv[tid * 2]);
        if (tid * 2 < HEAD_DIM / 2) {
            q_rope_1 = *(half2*)(&local_q[HEAD_DIM / 2 + tid * 2]);
            k_rope_1 = *(half2*)(&local_kv[HEAD_DIM / 2 + tid * 2]);
            cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
            sin_reg = {-sin[HEAD_DIM / 2 + tid * 2], -sin[HEAD_DIM / 2 + tid * 2 + 1]};
        } else {
            q_rope_1 = *(half2*)(&local_q[tid * 2 - HEAD_DIM / 2]);
            k_rope_1 = *(half2*)(&local_kv[tid * 2 - HEAD_DIM / 2]);
            cos_reg = {cos[tid * 2], cos[tid * 2 + 1]};
            sin_reg = {sin[tid * 2 - HEAD_DIM / 2], sin[tid * 2 + 1 - HEAD_DIM / 2]};
        }
        *(half2*)(&local_q[tid * 2]) = __hadd2(__hmul2(q_rope, __float22half2_rn(cos_reg)), __hmul2(q_rope_1, __float22half2_rn(sin_reg)));
        *(half2*)(&local_kv[tid * 2]) = __hadd2(__hmul2(k_rope, __float22half2_rn(cos_reg)), __hmul2(k_rope_1, __float22half2_rn(sin_reg)));
    }
    block.sync();

    uint32_t size = HEAD_DIM * sizeof(half);
    __shared__ uint64_t barrier;
    if (tid == 0) {
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            :
            : "r"(bar_ptr), "r"(1)
        );
        asm volatile (
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
            :
            : "r"(bar_ptr), "r"(size)
        );
    }
    cluster.sync();

    // Q reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        // Load neighbor block shmem data to this block's buffer within cluster
        if (tid == 0) {
            uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_q));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );

        // Add
        if (tid < HEAD_DIM / 2) {
            half2 buffer = *(half2*)(&local_buffer[tid * 2]);
            *(half2*)(&local_q[tid * 2]) = __hadd2(*(half2*)(&local_q[tid * 2]), buffer);
        }
        block.sync();
    }

    // K reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        // Load neighbor block shmem data to this block's buffer within cluster
        if (tid == 0) {
            uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_kv));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );

        // Add
        if (tid < HEAD_DIM / 2) {
            half2 buffer = *(half2*)(&local_buffer[tid * 2]);
            *(half2*)(&local_kv[tid * 2]) = __hadd2(*(half2*)(&local_kv[tid * 2]), buffer);
        }
        block.sync();
    }
 
    // Compute Q @ K^T
    __shared__ __align__(16) half attn_weight[KV_DIM_PER_BLOCK];
    uint kv_idx = head_id * HEAD_DIM * SEQ_LEN + cluster_block_id * KV_DIM_PER_BLOCK * HEAD_DIM + warp_id * NUM_ROW_PER_WARP_2 * HEAD_DIM + (lane_id / NUM_THREAD_PER_ROW) * HEAD_DIM + (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    for (int i = 0; i < NUM_ROW_PER_WARP_2; i+=NUM_ROW_PER_WARP) {
        tmp = 0.0;
        for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW) {
            *(uint4*)(&reg_input[0][0]) = *(uint4*)(&local_q[input_idx + j]);
            if (cluster_block_id == CLUSTER_SIZE - 1 && warp_id == NUM_WARPS - 1 && (i + lane_id / NUM_THREAD_PER_ROW) == NUM_ROW_PER_WARP_2 - 1) {
                *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&local_kv[input_idx + j]);
            } else {
                *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&k_cache[kv_idx + i * HEAD_DIM + j]);
            }
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
        if (lane_id % NUM_THREAD_PER_ROW == 0) {
            attn_weight[warp_id * NUM_ROW_PER_WARP_2 + i + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
        }
    }
    block.sync();
    
    // Softmax
    float local_scale = 0.0f;
    __shared__ float final_scale;
    for (int i = 0; i < KV_DIM_PER_BLOCK / BLOCK_SIZE; i++) {
        half tmp = hexp(attn_weight[tid * (KV_DIM_PER_BLOCK / BLOCK_SIZE) + i] / __float2half(1.0 * HEAD_DIM));
        attn_weight[i] = tmp;
        local_scale += __half2float(tmp);
    }
    __shared__ float local_attn_weight_reduction[16];
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_scale += __shfl_down_sync(0xffffffff, local_scale, mask);
    }
    if (lane_id == 0)
        local_attn_weight_reduction[warp_id] = local_scale;
    __syncthreads();
    if (tid < 16)
        local_scale = local_attn_weight_reduction[tid];
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        local_scale += __shfl_down_sync(0xffffffff, local_scale, mask);
    }
    if(tid == 0) {
        final_scale = local_scale;
    }
    // Reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_scale = final_scale;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            float* dst_shmem = cluster.map_shared_rank(&final_scale, dst_cta);
            atomicAdd(dst_shmem, local_scale);
        }
        cluster.sync();
    }
    for (int i = tid; i < KV_DIM_PER_BLOCK; i+=BLOCK_SIZE) {
        attn_weight[i] = __float2half(__half2float(attn_weight[i]) / final_scale);
    }
    block.sync();

    // Compute hidden @ wv
    tmp = 0.0;
    for (int i = 0; i < DIM_PER_BLOCK; i+=NUM_PER_ROW) { // 16
        *(uint4*)(&reg_input[0][0]) = *(uint4*)(&input_shmem[input_idx + i]);
        *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&w_v[weight_idx + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_kv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();
    
    // V reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_kv));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );

        // Add
        if (tid < HEAD_DIM / 2) {
            half2 buffer = *(half2*)(&local_buffer[tid * 2]);
            *(half2*)(&local_kv[tid * 2]) = __hadd2(*(half2*)(&local_kv[tid * 2]), buffer);
        }
        block.sync();
    }

    // Compute attn_weight @ V
    __shared__ __align__(16) half local_output[HEAD_DIM];
    kv_idx = head_id * HEAD_DIM * SEQ_LEN + cluster_block_id * KV_DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP * SEQ_LEN + (lane_id / NUM_THREAD_PER_ROW) * SEQ_LEN + (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;;
    tmp = 0.0;
    for (int i = 0; i < KV_DIM_PER_BLOCK; i+=NUM_PER_ROW) {
        *(uint4*)(&reg_input[0][0]) = *(uint4*)(&attn_weight[input_idx + i]);
        *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&v_cache[kv_idx + i]);
        if (cluster_block_id == CLUSTER_SIZE - 1 && i == KV_DIM_PER_BLOCK - NUM_PER_ROW && (lane_id % 2) == 1) {
            reg_weight[0][NUM_PER_THREAD - 1] = local_kv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW];
        }
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_output[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // output reduce throught DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_output));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );

        // Add
        if (tid < HEAD_DIM / 2) {
            half2 buffer = *(half2*)(&local_buffer[tid * 2]);
            *(half2*)(&local_output[tid * 2]) = __hadd2(*(half2*)(&local_output[tid * 2]), buffer);
        }
        block.sync();
    }

    // Compute output @ w_o
    weight_idx = head_id * HEAD_DIM * HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK * HEAD_DIM + warp_id * NUM_ROW_PER_WARP_2 * HEAD_DIM + (lane_id / NUM_THREAD_PER_ROW) * HEAD_DIM + (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    for (int i = 0; i < NUM_ROW_PER_WARP_2; i+=NUM_ROW_PER_WARP) {
        tmp = 0.0;
        for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW) {
            *(uint4*)(&reg_input[0][0]) = *(uint4*)(&local_output[input_idx + j]);
            *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&w_o[weight_idx + i * HEAD_DIM + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
        if (lane_id % NUM_THREAD_PER_ROW == 0) {
            atomicAdd(&global[cluster_block_id * DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP_2 + i + lane_id / NUM_THREAD_PER_ROW], __float2half(tmp));
        }
    }
    block.sync();
    
    // Fused residual and RMSNorm
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        *(half2*)(&reg_input_norm[0]) = __hadd2(*(half2*)(&input_shmem[d]), *(half2*)(&global[d]));
        for (int di = 0; di < 2; di++)
            local_sum += __half2float(reg_input_norm[di] * reg_input_norm[di]);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }
    if (lane_id == 0){
        norm_reduction[warp_id] = local_sum;
    }
    __syncthreads(); 
    if (tid < 16) 
        local_sum = norm_reduction[tid];
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    } 
    if (tid == 0)
        cluster_local_sum = local_sum;
    cluster.sync();
    // Reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            float* dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
            atomicAdd(dst_shmem, local_sum);
        }
        cluster.sync();
    }
    rms_rcp = __float2half(1.f / (std::sqrt(cluster_local_sum / float(HIDDEN_DIM)) + eps));
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
        *(half2*)(&reg_input_norm[0]) = __hmul2(*(half2*)(&reg_input_norm[0]), {rms_rcp, rms_rcp});
        *(half2*)(&reg_weight_norm[0]) = *(half2*)(&w_rms_attn[d]);
        *(half2*)(&input_shmem[d]) = __hmul2(*(half2*)(&reg_input_norm[0]), *(half2*)(&reg_weight_norm[0]));
    }
    block.sync();
    
    #pragma unroll
    for (int i = tid; i < DIM_PER_BLOCK; i+=BLOCK_SIZE) {
        input_shmem[i] = global[cluster_block_id * DIM_PER_BLOCK + i];
    }
    block.sync();

    // Compute gate proj
    weight_idx = head_id * HEAD_DIM * HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP * HIDDEN_DIM + (lane_id / NUM_THREAD_PER_ROW) * HIDDEN_DIM + (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    for (int i = 0; i < DIM_PER_BLOCK; i+=NUM_PER_ROW) { // 16
        *(uint4*)(&reg_input[0][0]) = *(uint4*)(&input_shmem[input_idx + i]);
        *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&ffn_gate[weight_idx + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_output[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // gate proj reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_output));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );

        // Add
        if (tid < HEAD_DIM / 2) {
            half2 buffer = *(half2*)(&local_buffer[tid * 2]);
            if (i == cluster.num_blocks() - 2) // ReLU
                *(half2*)(&local_output[tid * 2]) = __hmax2(__hadd2(*(half2*)(&local_output[tid * 2]), buffer), __float22half2_rn({0.0f, 0.0f}));
            else
                *(half2*)(&local_output[tid * 2]) = __hadd2(*(half2*)(&local_output[tid * 2]), buffer);
        }
        __syncthreads();
    }

    // Compute up proj
    __shared__ __align__(16) half local_output_up[HEAD_DIM];
    for (int i = 0; i < DIM_PER_BLOCK; i+=NUM_PER_ROW) { // 16
        *(uint4*)(&reg_input[0][0]) = *(uint4*)(&input_shmem[input_idx + i]);
        *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&ffn_up[weight_idx + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[0][d] * reg_weight[0][d]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_output_up[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // up proj reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_output_up));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );

        // Add
        if (tid < HEAD_DIM / 2) {
            half2 buffer = *(half2*)(&local_buffer[tid * 2]);
            *(half2*)(&local_output_up[tid * 2]) = __hadd2(*(half2*)(&local_output_up[tid * 2]), buffer);
        }
        __syncthreads();
    }

    // Compute down proj and residual
    half __align__(16) reg_input_2[1][NUM_PER_THREAD];
    weight_idx = head_id * HEAD_DIM * HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK * HEAD_DIM + warp_id * NUM_ROW_PER_WARP_2 * HEAD_DIM + (lane_id / NUM_THREAD_PER_ROW) * HEAD_DIM + (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    for (int i = 0; i < NUM_ROW_PER_WARP_2; i+=NUM_ROW_PER_WARP) {
        tmp = 0.0;
        for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW) {
            *(uint4*)(&reg_input[0][0]) = *(uint4*)(&local_output[input_idx + j]);
            *(uint4*)(&reg_input_2[0][0]) = *(uint4*)(&local_output_up[input_idx + j]);
            *(uint4*)(&reg_weight[0][0]) = *(uint4*)(&ffn_down[weight_idx + i * HEAD_DIM + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[0][d] * reg_input_2[0][d] * reg_weight[0][d]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
        if (lane_id % NUM_THREAD_PER_ROW == 0) {
            atomicAdd(&output[cluster_block_id * DIM_PER_BLOCK + warp_id * NUM_ROW_PER_WARP_2 + i + lane_id / NUM_THREAD_PER_ROW], __float2half(tmp));
        }
    }
}

int main(int argc, char** argv) {
    cudaFuncSetAttribute(single_decode, cudaFuncAttributeNonPortableClusterSizeAllowed, 16);

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
    half *h_rms_input, *d_rms_input;
    half *h_rms_attn, *d_rms_attn;
    float *h_cos, *d_cos;
    float *h_sin, *d_sin;
    h_input = new half[1 * HIDDEN_DIM];
    h_w_q = new half[HEAD_NUM * HIDDEN_DIM * HEAD_DIM];
    h_w_k = new half[HEAD_NUM * HIDDEN_DIM * HEAD_DIM];
    h_w_v = new half[HEAD_NUM * HIDDEN_DIM * HEAD_DIM];
    h_w_o = new half[HEAD_NUM * HEAD_DIM * HIDDEN_DIM];
    h_k_cache = new half[HEAD_NUM * SEQ_LEN * HEAD_DIM];
    h_v_cache = new half[HEAD_NUM * SEQ_LEN * HEAD_DIM];
    h_ffn_gate = new half[HIDDEN_DIM * FFN_HIDDEN];
    h_ffn_up = new half[HIDDEN_DIM * FFN_HIDDEN];
    h_ffn_down = new half[FFN_HIDDEN * HIDDEN_DIM];
    h_rms_input = new half[HIDDEN_DIM];
    h_rms_attn = new half[HIDDEN_DIM];
    h_cos = new float[HEAD_DIM];
    h_sin = new float[HEAD_DIM];

    fill_matrix(h_input, 1 * HIDDEN_DIM);
    fill_matrix(h_w_q, HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    fill_matrix(h_w_k, HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    fill_matrix(h_w_v, HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    fill_matrix(h_w_o, HEAD_NUM * HEAD_DIM * HIDDEN_DIM);
    fill_matrix(h_k_cache, HEAD_NUM * SEQ_LEN * HEAD_DIM);
    fill_matrix(h_v_cache, HEAD_NUM * SEQ_LEN * HEAD_DIM);
    fill_matrix(h_ffn_gate, HIDDEN_DIM * FFN_HIDDEN);
    fill_matrix(h_ffn_down, FFN_HIDDEN * HIDDEN_DIM);
    fill_matrix(h_ffn_up, HIDDEN_DIM * FFN_HIDDEN);
    fill_matrix(h_rms_input, HIDDEN_DIM);
    fill_matrix(h_rms_attn, HIDDEN_DIM);

    // Init cos, sin used in RoPE
    int encode_point_offset = 0;
    float rope_scale = 1;
    float rope_theta = 500000;
    for (int j = 0; j < HEAD_DIM; j++) {
        float inv_freq =(encode_point_offset / rope_scale) / (std::pow(rope_theta, float(2 * (j % (HEAD_DIM / 2))) / float(HEAD_DIM)));
        h_cos[j] = std::cos(inv_freq);
        h_sin[j] = std::sin(inv_freq);
    }

    cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * 1 * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_q), sizeof(half) * HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_k), sizeof(half) * HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_v), sizeof(half) * HEAD_NUM * HIDDEN_DIM * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_o), sizeof(half) * HEAD_NUM * HEAD_DIM * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_k_cache), sizeof(half) * HEAD_NUM * SEQ_LEN * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_v_cache), sizeof(half) * HEAD_NUM * SEQ_LEN * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_gate), sizeof(half) * HIDDEN_DIM * FFN_HIDDEN);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_down), sizeof(half) * FFN_HIDDEN * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_ffn_up), sizeof(half) * HIDDEN_DIM * FFN_HIDDEN);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_input), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_rms_attn), sizeof(half) * HIDDEN_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_cos), sizeof(float) * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_sin), sizeof(float) * HEAD_DIM);

    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_q), h_w_q, sizeof(half) * HEAD_NUM * HIDDEN_DIM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_k), h_w_k, sizeof(half) * HEAD_NUM * HIDDEN_DIM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_v), h_w_v, sizeof(half) * HEAD_NUM * HIDDEN_DIM * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_o), h_w_o, sizeof(half) * HEAD_NUM * HEAD_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_k_cache), h_k_cache, sizeof(half) * HEAD_NUM * SEQ_LEN * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_v_cache), h_v_cache, sizeof(half) * HEAD_NUM * SEQ_LEN * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_gate), h_ffn_gate, sizeof(half) * HIDDEN_DIM * FFN_HIDDEN, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_down), h_ffn_down, sizeof(half) * FFN_HIDDEN * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_ffn_up), h_ffn_up, sizeof(half) * HIDDEN_DIM * FFN_HIDDEN, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_input), h_rms_input, sizeof(half) * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_rms_attn), h_rms_attn, sizeof(half) * HIDDEN_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_cos), h_cos, sizeof(float) * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_sin), h_sin, sizeof(float) * HEAD_DIM, cudaMemcpyHostToDevice);

    half* h_output, *d_output;
    h_output = new half[1 * HIDDEN_DIM];
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * 1 * HIDDEN_DIM);

    half *global_reduce;
    cudaMalloc(reinterpret_cast<void**>(&global_reduce), sizeof(half) * HIDDEN_DIM);

    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    int wmup = 500;
    int test = 10;
    cudaEvent_t st, ed;
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    for (int i = 0; i < wmup; i++) {
        single_decode<<<grid, block>>>(
            d_output,
            d_input,
            d_w_q,
            d_w_k,
            d_w_v,
            d_w_o,
            d_k_cache,
            d_v_cache,
            d_ffn_gate,
            d_ffn_down,
            d_ffn_up,
            global_reduce,
            d_rms_input,
            d_rms_attn,
            d_cos,
            d_sin
        );
    }
    // cudaEventRecord(st);
    for (int i = 0; i < test; i++) {
        cudaEventRecord(st);
        single_decode<<<grid, block>>>(
            d_output,
            d_input,
            d_w_q,
            d_w_k,
            d_w_v,
            d_w_o,
            d_k_cache,
            d_v_cache,
            d_ffn_gate,
            d_ffn_down,
            d_ffn_up,
            global_reduce,
            d_rms_input,
            d_rms_attn,
            d_cos,
            d_sin
        );
        cudaEventRecord(ed);
        cudaEventSynchronize(ed);
        float ms;
        cudaEventElapsedTime(&ms, st, ed);
        std::cout << "Latency: " << ms * 1e3 << " us" << std::endl;
    }
    // cudaEventRecord(ed);
    // cudaEventSynchronize(ed);
    // float ms;
    // cudaEventElapsedTime(&ms, st, ed);
    // std::cout << "Latency: " << (ms / (1.0 * test)) * 1e3 << " us" << std::endl;
    cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * 1 * HIDDEN_DIM, cudaMemcpyDeviceToHost);
    return 0;
}