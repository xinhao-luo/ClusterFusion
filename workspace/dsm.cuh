#ifndef DSM_CUH
#define DSM_CUH

#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <cuda.h>    

namespace cg = cooperative_groups;

enum class Stage {
    LINEAR,
    ATTN,
    FFN,
    LINEAR_DEEPSEEK,
    QUK_DEEPSEEK,
    ATTN_DEEPSEEK
};

template <int cluster_size, Stage stage>
__device__ __forceinline__ void __cluster_dims__(cluster_size, 1, 1) dsm_ring_allreduce(
    const uint32_t size, const uint32_t tid, const uint32_t tile_size, 
    const uint32_t cluster_block_id, const uint32_t src_addr, const uint32_t dst_addr, 
    uint32_t barrier, uint32_t neighbor_dst_bar, half* src, half* dst
) {
    cg::cluster_group cluster = cg::this_cluster();
    uint32_t dst_cta, neighbor_dst_addr;
    half2 buffer;
    half __align__(16) reg_input[8];

    auto cluster_bar_ptr = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(
        static_cast<uintptr_t>(barrier)
    );

    if constexpr (stage == Stage::QUK_DEEPSEEK) {
        for (int i = 1; i < cluster.num_blocks(); i++) {
            if (tid == 0) {
                // initialize the cluster transaction barrier for this CTA
                // (equivalent to mbarrier.init.shared::cta.b64)
                cluster_bar_ptr->init(1);

                // announce arrival and expected transaction bytes
                // (equivalent to mbarrier.arrive.expect_tx.shared::cta.b64)
                // CUTLASS API: arrive_and_expect_tx(transaction_bytes)
                cluster_bar_ptr->arrive_and_expect_tx(size);
            }
            cluster.sync();

            if (tid == 0) {
                dst_cta = (cluster_block_id + i) % cluster.num_blocks();
                asm volatile (
                    "mapa.shared::cluster.u32 %0, %1, %2;\n"
                    : "=r"(neighbor_dst_addr)
                    : "r"(dst_addr), "r"(dst_cta)
                );
                asm volatile (
                    "mapa.shared::cluster.u32 %0, %1, %2;\n"
                    : "=r"(neighbor_dst_bar)
                    : "r"(barrier), "r"(dst_cta)
                );
                asm volatile (
                    "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                    :
                    :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                    : "memory"
                );
            }

            // wait for the transaction to complete.
            // CUTLASS often exposes a try_wait() or wait() method; prefer wait() if available.
            // Here we use a try_wait() spin (semantic same as original try_wait/parity loop).
            while (!cluster_bar_ptr->try_wait(0)) {
                // busy-spin until complete (matches original busy-wait behavior).
                // If cutlass offers a blocking wait(token) prefer it for lower spinning cost:
                //    cluster_bar_ptr->wait(token);
                // If your CUTLASS version uses different names, replace try_wait() accordingly.
            }

            cluster.sync();
        }
        return;
    } else {
        for (int i = 1; i < cluster.num_blocks() - 1; i++) {
            if (tid == 0) {
                cluster_bar_ptr->init(1);
                cluster_bar_ptr->arrive_and_expect_tx(size);
            }
            cluster.sync();

            if (tid == 0) {
                dst_cta = (cluster_block_id + i) % cluster.num_blocks();
                asm volatile (
                    "mapa.shared::cluster.u32 %0, %1, %2;\n"
                    : "=r"(neighbor_dst_addr)
                    : "r"(dst_addr), "r"(dst_cta)
                );
                asm volatile (
                    "mapa.shared::cluster.u32 %0, %1, %2;\n"
                    : "=r"(neighbor_dst_bar)
                    : "r"(barrier), "r"(dst_cta)
                );
                asm volatile (
                    "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                    :
                    :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                    : "memory"
                );
            }

            // wait for the local barrier's transaction to complete.
            while (!cluster_bar_ptr->try_wait(0)) {
                // spin
            }

            // Local reduce-add (unchanged per stage)
            if constexpr (stage == Stage::LINEAR) {
                if (tid < tile_size / 2) {
                    for (int j = 0; j < 3; j++) {
                        buffer = *(half2*)(&dst[j * tile_size + tid * 2]);
                        *(half2*)(&src[j * tile_size + tid * 2]) = __hadd2(*(half2*)(&src[j * tile_size + tid * 2]), buffer);
                    }
                }
            } else if constexpr (stage == Stage::ATTN) {
                if (tid < tile_size / 2) {
                    buffer = *(half2*)(&dst[tid * 2]);
                    *(half2*)(&src[tid * 2]) = __hadd2(*(half2*)(&src[tid * 2]), buffer);
                }
            } else if constexpr (stage == Stage::FFN) {
                if (tid < tile_size / 2) {
                    for (int j = 0; j < 3; j++) {
                        buffer = *(half2*)(&dst[j * tile_size + tid * 2]);
                        if (i == cluster.num_blocks() - 2) // ReLU
                            *(half2*)(&src[j * tile_size + tid * 2]) = __hmax2(__hadd2(*(half2*)(&src[j * tile_size + tid * 2]), buffer), __float22half2_rn({0.0f, 0.0f}));
                        else
                            *(half2*)(&src[j * tile_size + tid * 2]) = __hadd2(*(half2*)(&src[j * tile_size + tid * 2]), buffer);
                    }
                    for (int j = 0; j < 3; j++) {
                        buffer = *(half2*)(&dst[tile_size * 3 + j * tile_size + tid * 2]);
                        *(half2*)(&src[tile_size * 3 + j * tile_size + tid * 2]) = __hadd2(*(half2*)(&src[tile_size * 3 + j * tile_size + tid * 2]), buffer);
                    }
                }
            } else if constexpr (stage == Stage::LINEAR_DEEPSEEK) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&dst[tid * 8]);
                for (int di = 0; di < 8; di++) 
                    src[tid * 8 + di] = __hadd(src[tid * 8 + di], reg_input[di]);
                src[tile_size + tid] = __hadd(src[tile_size + tid], dst[tile_size + tid]);
            } else if constexpr (stage == Stage::ATTN_DEEPSEEK) {
                if (tid < tile_size / 8) {
                    *(uint4*)(&reg_input[0]) = *(uint4*)(&dst[tid * 8]);
                    for (int di = 0; di < 8; di++) 
                        src[tid * 8 + di] = __hadd(src[tid * 8 + di], reg_input[di]);
                }
            } else
                assert(false && "Unknown stage");
    
            cluster.sync();
        }
    }
}



#endif // DSM_CUH