#include "cuda_runtime.h"                
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <cuda.h>    

namespace cg = cooperative_groups;

#define BLOCK_SIZE (32 * 4)

enum class Stage {
    REDUCE,
    GATHER
};

template <int cluster_size, Stage stage>
__device__ __forceinline__ void __cluster_dims__(cluster_size, 1, 1) dsm_ring_allreduce(
    const uint32_t size, const uint32_t tid, const uint32_t tile_size, 
    const uint32_t cluster_block_id, const uint32_t src_addr, const uint32_t dst_addr, 
    uint32_t barrier, uint32_t neighbor_dst_bar, half* src, half* dst
) {
    cg::cluster_group cluster = cg::this_cluster();
    uint32_t dst_cta, neighbor_dst_addr;
    half __align__(16) reg_input[8];

    if constexpr (stage == Stage::GATHER) {
        for (int i = 1; i < cluster.num_blocks(); i++) {
            if (tid == 0) {
                asm volatile (
                    "mbarrier.init.shared::cta.b64 [%0], %1;"
                    :
                    : "r"(barrier), "r"(1)
                );
                asm volatile (
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                    :
                    : "r"(barrier), "r"(size)
                );
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
            asm volatile (
                "{\n"
                ".reg .pred                P1;\n"
                "LAB_WAIT:\n"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                "@P1                       bra.uni DONE;\n"
                "bra.uni                   LAB_WAIT;\n"
                "DONE:\n"
                "}\n"
                :: "r"(barrier),
                "r"(0)
            );
            cluster.sync();
        }
        return;
    } else {
        for (int i = 1; i < cluster.num_blocks() - 1; i++) {
            if (tid == 0) {
                asm volatile (
                    "mbarrier.init.shared::cta.b64 [%0], %1;"
                    :
                    : "r"(barrier), "r"(1)
                );
                asm volatile (
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                    :
                    : "r"(barrier), "r"(size)
                );
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
            asm volatile (
                "{\n"
                ".reg .pred                P1;\n"
                "LAB_WAIT:\n"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                "@P1                       bra.uni DONE;\n"
                "bra.uni                   LAB_WAIT;\n"
                "DONE:\n"
                "}\n"
                :: "r"(barrier),
                "r"(0)
            );

            // Local reduce-add
            #pragma unroll
            for (int i = tid * 8; i < tile_size; i+=BLOCK_SIZE * 8) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&dst[i]);
                #pragma unroll
                for (int di = 0; di < 8; di++) 
                    src[i + di] += reg_input[di];
            }
            // if (tid < tile_size / 2) {
            //     buffer = *(half2*)(&dst[tid * 2]);
            //     *(half2*)(&src[tid * 2]) = __hadd2(*(half2*)(&src[tid * 2]), buffer);
            // }
            cluster.sync();
        }
    }
}