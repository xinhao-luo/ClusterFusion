#include <random>

#define HEAD_DIM 128    
#define HEAD_NUM 128     
#define FFN_DIM 12288   
#define HIDDEN_DIM 4096 
#define SEQ_LEN 1024
#define BATCH_SIZE 1

#define NUM_WARPS 4 // 4 8 16 32
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE) 
#define CLUSTER_SIZE 16 // 2 4 8 16
#define NUM_PER_THREAD 8
#define NUM_ROW_PER_WARP (HEAD_DIM / NUM_WARPS) 
#define NUM_THREAD_PER_ROW (WARP_SIZE / NUM_ROW_PER_WARP) 
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW) 
#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)
#define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) 
#define FFN_DIM_PER_CLUSTER (FFN_DIM / HEAD_NUM) 
#define MAX_SMEM_DIM FFN_DIM_PER_CLUSTER

#define TMA_LOAD_ONCE 64 // 8 16 32 64 128 256
#define TMA_LOAD_ONCE_MAX 256
#define TMA_LOAD_ONCE_NUM (TMA_LOAD_ONCE * HEAD_DIM)
#define TMA_LOAD_ONCE_SIZE (TMA_LOAD_ONCE_NUM * sizeof(half))
#define TMA_LOAD_ONCE_ATTN (TMA_LOAD_ONCE / 2)
#define TMA_LOAD_ONCE_NUM_ATTN ((TMA_LOAD_ONCE * HEAD_DIM) / 2)
#define TMA_LOAD_ONCE_SIZE_ATTN (TMA_LOAD_ONCE_NUM_ATTN * sizeof(half))
#define TMA_LOAD_ONCE_NUM_FFN (TMA_LOAD_ONCE * TMA_LOAD_ONCE_MAX)
#define TMA_LOAD_ONCE_NUM_FFN_TOTAL (TMA_LOAD_ONCE * FFN_DIM_PER_CLUSTER)
#define TMA_LOAD_ONCE_SIZE_FFN (TMA_LOAD_ONCE_NUM_FFN_TOTAL * sizeof(half))

#define NUM_THREAD_PER_ROW_2 (HEAD_DIM / NUM_PER_THREAD) // 16
#define NUM_ROW_PER_WARP_2 (WARP_SIZE / NUM_THREAD_PER_ROW_2) // 2
#define NUM_PER_ROW_2 (NUM_WARPS * NUM_ROW_PER_WARP_2) // 8
#define DEC_TILE (TMA_LOAD_ONCE_ATTN / NUM_PER_ROW_2)
#define NUM_ROW_PER_WARP_3 (TMA_LOAD_ONCE / NUM_WARPS) 
#define NUM_THREAD_PER_ROW_3 (WARP_SIZE / NUM_ROW_PER_WARP_3) 
#define NUM_PER_ROW_3 (NUM_PER_THREAD * NUM_THREAD_PER_ROW_3) 

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 0.1);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(0.01f);
        }   
    }   
}