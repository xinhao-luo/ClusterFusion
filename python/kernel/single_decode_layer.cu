#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <cuda.h>    
#include <cuda/barrier>
#include <cudaTypedefs.h>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
#include <iostream>
#include <stdio.h>
#include <torch/extension.h>

#define HEAD_DIM 128    // attn head dimension
#define HEAD_NUM 32     // attn head number
#define FFN_DIM 4096 // ffn hidden dimension
#define HIDDEN_DIM 4096 // token embedding dimension
#define SEQ_LEN 4096    // sequence length

#define NUM_WARPS 4 // 4 8 16 32
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE) 
#define CLUSTER_SIZE 4 // 2 4 8 16
#define NUM_PER_THREAD 8
#define NUM_ROW_PER_WARP (HEAD_DIM / NUM_WARPS) 
#define NUM_THREAD_PER_ROW (WARP_SIZE / NUM_ROW_PER_WARP) 
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW) 
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)
#define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) 

#define TMA_LOAD_ONCE 64 // 8 16 32 64 128 256
#define NUM_ROW_PER_WARP_2 (TMA_LOAD_ONCE / NUM_WARPS) 
#define NUM_THREAD_PER_ROW_2 (WARP_SIZE / NUM_ROW_PER_WARP_2) 
#define NUM_PER_ROW_2 (NUM_PER_THREAD * NUM_THREAD_PER_ROW_2) 

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) single_decode_kernel(
    half* output, // 1 * hidden_dim
    half* input,  // 1 * hidden_dim
    half* global_reduce,    // hidden_dim  
    half* w_rms_input,// hidden_dim
    half* w_rms_attn, // hidden_dim
    float* cos,       // head_dim
    float* sin,       // head_dim
    const __grid_constant__ CUtensorMap tensor_map, // 3 * hidden_dim * hidden_dim
    const __grid_constant__ CUtensorMap tensor_map_kv_cache, // 2 * seqlen * head_num * head_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_o, // hidden_dim * hidden_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_gate_up, // 2 * hidden_dim * ffn_dim
    const __grid_constant__ CUtensorMap tensor_map_weight_down // ffn_dim * hidden_dim
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

    // Init shared memory
    __shared__ __align__(16) half input_shmem[DIM_PER_BLOCK];
    __shared__ float reduction[NUM_WARPS];
    __shared__ float cluster_local_sum;
    __shared__ __align__(16) half local_q[HEAD_DIM];
    __shared__ __align__(16) half local_kv[HEAD_DIM];
    __shared__ alignas(128) half weight[TMA_LOAD_ONCE * HEAD_DIM];
    __shared__ alignas(128) half weight_buffer[TMA_LOAD_ONCE * HEAD_DIM];
    __shared__ __align__(16) half local_buffer[HEAD_DIM];
    __shared__ __align__(16) half attn_weight[KV_DIM_PER_BLOCK];
    // Initialize shared memory barrier with the number of threads participating in the barrier.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    __shared__ barrier bar_buffer;
    __shared__ uint64_t barrier;

    // Load input [1 x HIDDEN_DIM / CLUSTR_SIZE] to shared memory
    #pragma unroll
    for (int i = tid * 8; i < DIM_PER_BLOCK; i+=BLOCK_SIZE * 8) {
        *(uint4*)(&input_shmem[i]) = *(uint4*)(&input[cluster_block_id * DIM_PER_BLOCK + i]);
    }
    block.sync();

    // RMSNorm
    float local_sum = 0;
    half __align__(16) reg_input_norm[2], reg_weight_norm[2];
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        *(half2*)(&reg_input_norm[0]) = *(half2*)(&input_shmem[d]);
        for (int di = 0; di < 2; di++)
            local_sum += __half2float(__hmul(reg_input_norm[di], reg_input_norm[di]));
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }
    if (lane_id == 0){
        reduction[warp_id] = local_sum;
    }
    block.sync(); 
    if (tid < NUM_WARPS) 
        local_sum = reduction[tid];
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
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

    // Compute qkv_proj
    float tmp = 0.0;
    half __align__(16) reg_input[NUM_PER_THREAD];
    half __align__(16) reg_weight[NUM_PER_THREAD];
    if (tid == 0) {
        // Initialize barrier. All `blockDim.x` threads in block participate.
        init(&bar, blockDim.x);
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
        init(&bar_buffer, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    // Syncthreads so initialized barrier is visible to all threads.
    block.sync();

    // Preload weight_q
    barrier::arrival_token token[2];
    if (tid == 0) {
        // Initiate bulk tensor copy.
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK, bar);
        // Arrive on the barrier and tell how many bytes are expected to come in.
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        // Other threads just arrive.
        token[0] = bar.arrive();
    }

    // Compute input @ w_q
    uint input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        }
    }
    bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_q[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // Preload weight_k
    tmp = 0.0;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    // Compute input @ w_k
    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        }
    }
    bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
        }
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

    uint32_t size = HEAD_DIM * sizeof(half);
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));

    // Q reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        // Load neighbor block shmem data to this block's buffer within cluster
        if (tid == 0) {
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
        if (tid == 0) {
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
        cluster.sync();
    }
    
    // K reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        // Load neighbor block shmem data to this block's buffer within cluster
        if (tid == 0) {
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
        if (tid == 0) {
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
        cluster.sync();
    }

    // Compute Q @ K^T
    input_idx = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    weight_idx = warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2;
    tmp = 0.0;

    // Preload k_cache
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_kv_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map_kv_cache, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            tmp = 0.0;
            for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
                *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[weight_idx * HEAD_DIM + input_idx + j]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], reg_weight[d]));
                }
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                tmp += __shfl_down_sync(0xffffffff, tmp, mask);
            }
            if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
                attn_weight[warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2 + (id - 1) * TMA_LOAD_ONCE] = __float2half(tmp);
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_kv_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            tmp = 0.0;
            for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
                *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight_buffer[weight_idx * HEAD_DIM + input_idx + j]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], reg_weight[d]));
                }
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                tmp += __shfl_down_sync(0xffffffff, tmp, mask);
            }
            if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
                attn_weight[warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2 + (id - 1) * TMA_LOAD_ONCE] = __float2half(tmp);
            }
        }
    }
    bar_buffer.wait(std::move(token[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    tmp = 0.0;
    for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
        if (cluster_block_id == CLUSTER_SIZE - 1 && warp_id == NUM_WARPS - 1 && lane_id / NUM_THREAD_PER_ROW_2 == NUM_ROW_PER_WARP_2 - 1) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_kv[input_idx + j]);
        } else {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight_buffer[weight_idx * HEAD_DIM + input_idx + j]);
        }
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], reg_weight[d]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        attn_weight[warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2 + (KV_DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) * TMA_LOAD_ONCE] = __float2half(tmp);
    }
    block.sync();

    // Softmax
    float local_scale = 0.0f;
    half tmp_;
    for (int i = 0; i < KV_DIM_PER_BLOCK / BLOCK_SIZE; i++) {
        tmp_ = hexp(__hdiv(attn_weight[tid * (KV_DIM_PER_BLOCK / BLOCK_SIZE) + i], __float2half(sqrt(1.0 * HEAD_DIM))));
        attn_weight[tid * (KV_DIM_PER_BLOCK / BLOCK_SIZE) + i] = tmp_;
        local_scale += __half2float(tmp_);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_scale += __shfl_down_sync(0xffffffff, local_scale, mask);
    }
    if (lane_id == 0) {
        reduction[warp_id] = local_scale;
    }
    block.sync();
    if (tid < NUM_WARPS) {
        local_scale = reduction[tid];
    }
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
        local_scale += __shfl_down_sync(0xffffffff, local_scale, mask);
    }
    if(tid == 0) {
        cluster_local_sum = local_scale;
    }
    cluster.sync();
    // Reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_scale = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            float* dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);
            atomicAdd(dst_shmem, local_scale);
        }
        cluster.sync();
    }
    for (int i = tid; i < KV_DIM_PER_BLOCK; i+=BLOCK_SIZE) {
        attn_weight[i] = __float2half(__half2float(attn_weight[i]) / cluster_local_sum);
    }
    block.sync();
    
    input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    // Preload weight_v
    tmp = 0.0;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM * 2 + cluster_block_id * DIM_PER_BLOCK, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    // Compute input @ w_v
    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM * 2 + cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map, head_id * HEAD_DIM, HIDDEN_DIM * 2 + cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        }
    }
    bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_kv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }

    // V reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
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
        if (tid == 0) {
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
        cluster.sync();
    }
    
    // Compute attn_weight @ V
    // Preload V
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_kv_cache, head_id * HEAD_DIM, SEQ_LEN + cluster_block_id * KV_DIM_PER_BLOCK, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    tmp = 0.0;
    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map_kv_cache, head_id * HEAD_DIM, SEQ_LEN + cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&attn_weight[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_kv_cache, head_id * HEAD_DIM, SEQ_LEN + cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&attn_weight[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        }
    }
    bar_buffer.wait(std::move(token[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&attn_weight[input_idx + ((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        if (cluster_block_id == CLUSTER_SIZE - 1 && i == TMA_LOAD_ONCE - NUM_PER_ROW && (lane_id % NUM_THREAD_PER_ROW) == NUM_THREAD_PER_ROW - 1) {
            weight_buffer[(input_idx + i + NUM_PER_THREAD - 1) * HEAD_DIM + weight_idx] = local_kv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW];
        }
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_q[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }

    // output reduce throught DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
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
        if (tid == 0) {
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
        cluster.sync();
    }

    // Compute output @ w_o
    input_idx = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    weight_idx = warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_weight_o, cluster_block_id * DIM_PER_BLOCK, head_id * HEAD_DIM, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map_weight_o, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, head_id * HEAD_DIM, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            tmp = 0.0;
            for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight[(input_idx + j + d) * TMA_LOAD_ONCE + weight_idx]));
                }
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                tmp += __shfl_down_sync(0xffffffff, tmp, mask);
            }
            if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
                atomicAdd(&global_reduce[cluster_block_id * DIM_PER_BLOCK + weight_idx + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_weight_o, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, head_id * HEAD_DIM, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            tmp = 0.0;
            for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + j + d) * TMA_LOAD_ONCE + weight_idx]));
                }
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                tmp += __shfl_down_sync(0xffffffff, tmp, mask);
            }
            if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
                atomicAdd(&global_reduce[cluster_block_id * DIM_PER_BLOCK + weight_idx + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
            }
        }
    }
    bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    tmp = 0.0;
    for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + j + d) * TMA_LOAD_ONCE + weight_idx]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        atomicAdd(&global_reduce[cluster_block_id * DIM_PER_BLOCK + weight_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    }
    cluster.sync();

    // Fused residual and RMSNorm
    local_sum = 0.0;
    cluster_local_sum = 0.0;
    for (int d = tid * 2; d < DIM_PER_BLOCK; d+=BLOCK_SIZE * 2) { 
        *(half2*)(&reg_input_norm[0]) = __hadd2(*(half2*)(&input_shmem[d]), *(half2*)(&global_reduce[cluster_block_id * DIM_PER_BLOCK + d]));
        *(half2*)(&input_shmem[d]) = *(half2*)(&reg_input_norm[0]);
        for (int di = 0; di < 2; di++)
            local_sum += __half2float(__hmul(reg_input_norm[di], reg_input_norm[di]));
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }
    if (lane_id == 0){
        reduction[warp_id] = local_sum;
    }
    block.sync(); 
    if (tid < NUM_WARPS){
        local_sum = reduction[tid];
    }
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
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
    
    // Compute gate proj
    // Preload weight_gate
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_weight_gate_up, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    // Compute input @ ffn_gate
    tmp = 0.0;
    input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map_weight_gate_up, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_weight_gate_up, head_id * HEAD_DIM, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        }
    }
    bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_q[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();
    
    // Preload weight_up
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_weight_gate_up, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    // Compute input @ ffn_up
    tmp = 0.0;
    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map_weight_gate_up, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_weight_gate_up, head_id * HEAD_DIM, HIDDEN_DIM + cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
                }
            }
        }
    }
    bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    for (int i = 0; i < TMA_LOAD_ONCE; i+=NUM_PER_ROW) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(reg_input[d], weight_buffer[(input_idx + i + d) * HEAD_DIM + weight_idx]));
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0) {
        local_kv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);
    }
    block.sync();

    // gate proj reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
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
        if (tid == 0) {
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
                *(half2*)(&local_q[tid * 2]) = __hmax2(__hadd2(*(half2*)(&local_q[tid * 2]), buffer), __float22half2_rn({0.0f, 0.0f}));
            else
                *(half2*)(&local_q[tid * 2]) = __hadd2(*(half2*)(&local_q[tid * 2]), buffer);
        }
        cluster.sync();
    }

    // up proj reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
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
        if (tid == 0) {
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
        cluster.sync();
    }

    // Compute up_gate mul and down_proj
    half __align__(16) reg_input_2[NUM_PER_THREAD];
    input_idx = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    weight_idx = warp_id * NUM_ROW_PER_WARP_2 + lane_id / NUM_THREAD_PER_ROW_2;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_weight_down, cluster_block_id * DIM_PER_BLOCK, head_id * HEAD_DIM, bar);
        token[0] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
    } else {
        token[0] = bar.arrive();
    }

    // Compute up_gate mul and down_proj
    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (id % 2) {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight_buffer, &tensor_map_weight_down, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, head_id * HEAD_DIM, bar_buffer);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar_buffer, 1, sizeof(weight_buffer));
            } else {
                token[id % 2] = bar_buffer.arrive();
            }
            bar.wait(std::move(token[(id - 1) % 2]));
            tmp = 0.0;
            for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
                *(uint4*)(&reg_input_2[0]) = *(uint4*)(&local_kv[input_idx + j]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(__hmul(reg_input[d], reg_input_2[d]), weight[(input_idx + j + d) * TMA_LOAD_ONCE + weight_idx]));
                }
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                tmp += __shfl_down_sync(0xffffffff, tmp, mask);
            }
            if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
                atomicAdd(&output[cluster_block_id * DIM_PER_BLOCK + weight_idx + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
            }
        } else {
            if (tid == 0) {
                cde::cp_async_bulk_tensor_2d_global_to_shared(&weight, &tensor_map_weight_down, cluster_block_id * DIM_PER_BLOCK + id * TMA_LOAD_ONCE, head_id * HEAD_DIM, bar);
                token[id % 2] = cuda::device::barrier_arrive_tx(bar, 1, sizeof(weight));
            } else {
                token[id % 2] = bar.arrive();
            }
            bar_buffer.wait(std::move(token[(id - 1) % 2]));
            tmp = 0.0;
            for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
                *(uint4*)(&reg_input_2[0]) = *(uint4*)(&local_kv[input_idx + j]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(__hmul(__hmul(reg_input[d], reg_input_2[d]), weight[(input_idx + j + d) * TMA_LOAD_ONCE + weight_idx]));
                }
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                tmp += __shfl_down_sync(0xffffffff, tmp, mask);
            }
            if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
                atomicAdd(&output[cluster_block_id * DIM_PER_BLOCK + weight_idx + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
            }
        }
    }
    bar_buffer.wait(std::move(token[((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) % 2]));
    tmp = 0.0;
    for (int j = 0; j < HEAD_DIM; j+=NUM_PER_ROW_2) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_q[input_idx + j]);
        *(uint4*)(&reg_input_2[0]) = *(uint4*)(&local_kv[input_idx + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(__hmul(__hmul(reg_input[d], reg_input_2[d]), weight[(input_idx + j + d) * TMA_LOAD_ONCE + weight_idx]));
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        atomicAdd(&output[cluster_block_id * DIM_PER_BLOCK + weight_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    }
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
) 
{
    cudaFuncSetAttribute(single_decode_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({1, HIDDEN_DIM}, 0, options);
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());
    half *reduce_workspace;
    cudaMalloc(reinterpret_cast<void**>(&reduce_workspace), sizeof(half) * 1 * HIDDEN_DIM);

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
    
    CUtensorMap tensor_map_weight{};
    CUtensorMap tensor_map_kv_cache{};
    CUtensorMap tensor_map_weight_o{};
    CUtensorMap tensor_map_weight_gate_up{};
    CUtensorMap tensor_map_weight_down{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {HIDDEN_DIM, 3 * HIDDEN_DIM};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map_weight,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       // cuuint32_t tensorRank,
        weight_qkv_ptr,                 // void *globalAddress,
        size,                       // const cuuint64_t *globalDim,
        stride,                     // const cuuint64_t *globalStrides,
        box_size,                   // const cuuint32_t *boxDim,
        elem_stride,                // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_kv_cache[rank] = {HIDDEN_DIM, 2 * SEQ_LEN};
    uint64_t stride_kv_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_kv_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    uint32_t elem_stride_kv_cache[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_kv_cache = cuTensorMapEncodeTiled(
        &tensor_map_kv_cache,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        kv_cache_ptr,                 
        size_kv_cache,                       
        stride_kv_cache,                     
        box_size_kv_cache,                   
        elem_stride_kv_cache,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_weight_o[rank] = {HIDDEN_DIM, HIDDEN_DIM};
    uint64_t stride_weight_o[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_o[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_weight_o[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_weight_o = cuTensorMapEncodeTiled(
        &tensor_map_weight_o,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        weight_o_ptr,                 
        size_weight_o,                       
        stride_weight_o,                     
        box_size_weight_o,                   
        elem_stride_weight_o,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_weight_gate_up[rank] = {FFN_DIM, 2 * HIDDEN_DIM};
    uint64_t stride_weight_gate_up[rank - 1] = {FFN_DIM * sizeof(half)};
    uint32_t box_size_weight_gate_up[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    uint32_t elem_stride_weight_gate_up[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_weight_gate_up = cuTensorMapEncodeTiled(
        &tensor_map_weight_gate_up,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        gate_up_proj_weight_ptr,                 
        size_weight_gate_up,                       
        stride_weight_gate_up,                     
        box_size_weight_gate_up,                   
        elem_stride_weight_gate_up,               
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    uint64_t size_weight_down[rank] = {HIDDEN_DIM, FFN_DIM};
    uint64_t stride_weight_down[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_down[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_weight_down[rank] = {1, 1};

    // Create the tensor descriptor.
    CUresult res_weight_down = cuTensorMapEncodeTiled(
        &tensor_map_weight_down,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        down_proj_weight_ptr,                 
        size_weight_down,                       
        stride_weight_down,                     
        box_size_weight_down,                   
        elem_stride_weight_down,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    single_decode_kernel<<<grid, block>>>(
        o_ptr,
        input_ptr,
        reduce_workspace,
        rms_input_weight_ptr,
        rms_attn_weight_ptr,
        cos_ptr,
        sin_ptr,
        tensor_map_weight,
        tensor_map_kv_cache,
        tensor_map_weight_o,
        tensor_map_weight_gate_up,
        tensor_map_weight_down
    );
    cudaDeviceSynchronize();
    return o;
}