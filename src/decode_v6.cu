#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cute/util/print.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "cuda_launch.hpp"
#include "shared_storage.h"
#include <random>  // C++标准库随机数生成器

#include <cute/layout.hpp>
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cooperative_groups.h"

// 使用模板定义，适应不同类型的Element
template <typename Element>
void initialize_matrix(thrust::host_vector<Element>& h_S, size_t M, size_t N, unsigned seed = 42) {
    // 使用 C++ 标准库的随机数生成器
    std::mt19937 rng(seed);  // 固定种子值，默认为42
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);  // 定义随机数分布

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            // 生成随机数并赋值到矩阵中，将浮点型转换为模板指定的Element类型
            h_S[m * N + n] = static_cast<Element>(dist(rng));
        }
    }
}

#define HEAD_DIM 128    // attn head dimension
#define HEAD_NUM 32     // attn head number
#define SEQ_LEN 4096    // sequence length

#define NUM_WARPS 8
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define CLUSTER_SIZE 4
#define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE)  

template <typename _TiledCopyS, typename _GmemLayout,
          typename _SmemLayout, typename _TileShape>
struct Params {
  using TiledCopyS = _TiledCopyS;
  using GmemLayout = _GmemLayout;
  using SmemLayout = _SmemLayout;
  using TileShape = _TileShape;

  TiledCopyS const tmaLoad;
  GmemLayout const gmemLayout;
  SmemLayout const smemLayout;
  TileShape const tileShape;

  Params(_TiledCopyS const &tmaLoad,
         _GmemLayout const &gmemLayout, _SmemLayout const &smemLayout,
         _TileShape const &tileShape)
      : tmaLoad(tmaLoad), gmemLayout(gmemLayout),
        smemLayout(smemLayout), tileShape(tileShape) {}
};


template <int kNumThreads, class Element, class Params>
__global__ static void __launch_bounds__(kNumThreads, 1)
    copyTMAKernel(CUTE_GRID_CONSTANT Params const params) {
  using namespace cute;

  //
  // Get layouts and tiled copies from Params struct
  //
  using GmemLayout = typename Params::GmemLayout;
  using SmemLayout = typename Params::SmemLayout;
  using TileShape = typename Params::TileShape;

  auto &tmaLoad = params.tmaLoad;
  auto &gmemLayout = params.gmemLayout;
  auto &smemLayout = params.smemLayout;
  auto &tileShape = params.tileShape;

  namespace cg = cooperative_groups;
  cg::grid_group grid             = cg::this_grid();
  cg::cluster_group cluster       = cg::this_cluster();
  cg::thread_block block          = cg::this_thread_block();
  const uint32_t head_id          = grid.cluster_rank() % HEAD_NUM;
  const uint32_t cluster_block_id = cluster.block_rank();
  const uint32_t tid              = block.thread_rank();

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTMA<Element, SmemLayout>;
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

  // Define smem tensor
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayout);

  // Get mbarrier object and its value type
  auto &mbarrier = shared_storage.mbarrier;
  using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;

  // Constants used for TMA
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const bool lane_predicate = cute::elect_one_sync();
  constexpr int kTmaTransactionBytes = sizeof(ArrayEngine<Element, size(SmemLayout{})>);

  // Prefetch TMA descriptors for load and store
  if (warp_idx == 0 && lane_predicate) {
    prefetch_tma_descriptor(tmaLoad.get_tma_descriptor());
  }

  // Get CTA view of gmem tensor
  Tensor mS = tmaLoad.get_tma_tensor(shape(gmemLayout));
  int k_cache_idx_0 = cluster_block_id * KV_DIM_PER_BLOCK;
  int k_cache_idx_1 = head_id * HEAD_DIM;
  auto blkCoord = make_coord(k_cache_idx_0, k_cache_idx_1);
  Tensor gS = local_tile(mS, tileShape, blkCoord);

  auto cta_tmaS = tmaLoad.get_slice(Int<0>{});

  if (warp_idx == 0 and lane_predicate) {
    mbarrier.init(1 /* arrive count */);
    mbarrier.arrive_and_expect_tx(kTmaTransactionBytes);
    copy(tmaLoad.with(reinterpret_cast<BarrierType &>(mbarrier)), cta_tmaS.partition_S(gS), cta_tmaS.partition_D(sS));
  }
  __syncthreads();

  mbarrier.wait(0 /* phase */);
  
  cutlass::arch::fence_view_async_shared();
  // block.sync();
  if(head_id == 0 && cluster_block_id == 0 && tid==0){
    print(tileShape);printf("  tileShape\n");
    printf("TileShape: (%d, %d)\n", static_cast<int>(size<0>(tileShape)),
       static_cast<int>(size<1>(tileShape)));
    print(sS); printf("  sS\n");
    print_tensor(sS);
  }

}

template <int TILE_M = 128, int TILE_N = 128, int THREADS = BLOCK_SIZE>
int copy_host_tma_load_and_store_kernel(int iterations = 1) {
  using namespace cute;

  using Element = cutlass::half_t;

  auto tensor_shape = make_shape(SEQ_LEN, HEAD_NUM * HEAD_DIM);

  // Allocate and initialize
  thrust::host_vector<Element> h_S(size(tensor_shape)); 
  // thrust::host_vector<Element> h_D(size(tensor_shape)); // (M, N)

  initialize_matrix(h_S, SEQ_LEN, HEAD_NUM * HEAD_DIM);

  thrust::device_vector<Element> d_S = h_S;
  // thrust::device_vector<Element> d_D = h_D;

  //
  // Make tensors
  //

  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  // auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), gmemLayoutS);
  // Tensor tensor_D = make_tensor(
  //     make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), gmemLayoutD);

  using bM = Int<TILE_M>;
  using bN = Int<TILE_N>;

  auto tileShape = make_shape(bM{}, bN{});
  // NOTE: same smem layout for TMA load and store

    // 定义 Shared Memory Layout Atom
    using SmemLayoutAtomO = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, 
        cutlass::half_t, 
        Int<TILE_M>, // 使用编译时常量
        Int<TILE_N>  // 使用编译时常量
        >());

    // 定义最终的 Shared Memory Layout
    using SmemLayoutO = decltype(
        tile_to_shape(SmemLayoutAtomO{}, make_shape(Int<TILE_M>{}, Int<TILE_N>{}))); // 保证类型兼容

//   auto smemLayout = SmemLayoutO{};
  auto smemLayout = make_layout(tileShape, LayoutRight{});
  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, tensor_S, smemLayout);
  // print(tma_load);

  // auto tma_store = make_tma_copy(SM90_TMA_STORE{}, tensor_D, smemLayout);
  // print(tma_store);

  Params params(tma_load, gmemLayoutS, smemLayout, tileShape);

  dim3 gridDim(HEAD_NUM * CLUSTER_SIZE);
  dim3 blockDim(BLOCK_SIZE);

  int smem_size = int(sizeof(SharedStorageTMA<Element, decltype(smemLayout)>));
  printf("smem size: %d.\n", smem_size);

  void const *kernel =
      (void const *)copyTMAKernel<BLOCK_SIZE, Element, decltype(params)>;
  cfk::utils::set_smem_size(smem_size, kernel);

  dim3 cluster_dims(CLUSTER_SIZE);

  // Define the cluster launch parameter structure.
  cutlass::ClusterLaunchParams launch_params{gridDim, blockDim, cluster_dims,
                                             smem_size};

  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();    
    cutlass::Status status =
        cutlass::launch_kernel_on_cluster(launch_params, kernel, params);
    cudaError result = cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    if (result != cudaSuccess) {
      std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
                << std::endl;
      return -1;
    }
    std::chrono::duration<double, std::milli> tDiff = t2 - t1;
    double time_ms = tDiff.count();
    // std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
    //           << 2e-6 * 128 * 128 * 128 * sizeof(Element) / time_ms << " GB/s)"
    //           << std::endl;
  }
    return 0;
}

int main(int argc, char const **argv) {
  int iterations = 1;
  copy_host_tma_load_and_store_kernel(iterations);
  return 0;
}
