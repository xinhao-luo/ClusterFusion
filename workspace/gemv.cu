#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "cuda_runtime_api.h"
#include <algorithm>
#include <stdint.h>

using namespace nvcuda;
using namespace std;

#define CUTLASS_ENABLE_L2_PREFETCH False 
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define UINT2(pointer) (reinterpret_cast<uint2*>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])

typedef enum{
    HGEMMAlignedV1,
    HGEMMAlignedV2,
    HGEMMAlignedV3,
    HGEMMAlignedV4,
    HGEMMAlignedV5,
    HGEMMAlignedV6
} F16F16GemmTCAlgo_t;

void cpuF16F16Gemm(half *a, half *b, half *c, int M, int N, int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += (float)a[OFFSET(m, k, K)] * (float)b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = (half)psum;
        }
    }
}

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define LDMATRIX_X1(R, addr) \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

#if ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#else
#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#endif

__device__ void LDG_L1_128bit_LAST(int4& dst, const uint8_t* ptr, bool pred_guard=true) \
{
    uint4 &data = reinterpret_cast<uint4 &>(dst);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
        "  @p ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
}



__device__ void LDG_L1_128bit_ALWAYS(int4& dst, const uint8_t* ptr, bool pred_guard=true) \
{
  uint4 &data = reinterpret_cast<uint4 &>(dst);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  mov.b32 %0, %6;\n"
        "  mov.b32 %1, %7;\n"
        "  mov.b32 %2, %8;\n"
        "  mov.b32 %3, %9;\n"
        "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        "}\n"
        : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
        : "l"(ptr), "r"((int)pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));

}

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4 
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

#define COLS_PER_WARP 1
#define COLS_PER_BLOCK 4  // COLS_PER_WARP * WARPS_PER_BLOCK
#define GROUP_SIZE 32       // WARP_SIZE / COLS_PER_WARP


#define NUM_PER_THREAD 8
#define ROW_THREAD 4                     // 4
#define COL_THREAD (WARP_SIZE / ROW_THREAD) // 8
#define COL_NUM_COUNT (COL_THREAD * NUM_PER_THREAD) // 64 half
#define ROW_NUM_COUNT (ROW_THREAD) // 4
#define BLOCK_ROW (ROW_THREAD * WARPS_PER_BLOCK) // 16

//#define K_TILE 32
//#if 128 >= COL_THREAD * NUM_PER_THREAD * 2 //for double-buffer
//    #define K_TILE 128 // COL_THREAD * NUM_PER_THREAD 
//#else
//    #define K_TILE (COL_THREAD * NUM_PER_THREAD * 4)
//#endif
#define K_TILE 128 

__global__ void mySGemvKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t K,
                                 size_t N){
    const size_t tid = threadIdx.x % WARP_SIZE;
    const size_t warpid = threadIdx.x / WARP_SIZE;
    const size_t block_row = blockIdx.x * WARPS_PER_BLOCK * ROW_NUM_COUNT;
    const size_t warp_row = warpid * ROW_NUM_COUNT;
    const size_t tid_row = tid / COL_THREAD;

    const size_t out_K_iters = K / K_TILE; // 32
    const size_t in_K_iters = K_TILE / COL_NUM_COUNT; // 2

    half tmp = 0.0;
    // int4 r_a;
    // int4 r_b;
    half __align__(16) r_a[2][8];
    half __align__(16) r_b[2][8];
    size_t A_idx = (tid % COL_THREAD) * 8; 
    size_t B_idx = (block_row + warp_row + tid_row) * N + (tid % COL_THREAD) * 8; 

    // for (size_t i = 0; i < out_K_iters; ++i){
    //     for (size_t j = 0; j < in_K_iters; ++j){
    //         *(uint4*)(&AS[A_idx + i * K_TILE + COL_NUM_COUNT * j]) = *(uint4*)(&A[A_idx + i * K_TILE + COL_NUM_COUNT * j]);
    //         *(uint4*)(&BS[B_idx + i * K_TILE + COL_NUM_COUNT * j]) = *(uint4*)(&B[B_idx + i * K_TILE + COL_NUM_COUNT * j]);
    //     }
    // }

    for (size_t i = 0; i < out_K_iters; ++i){
        for (size_t j = 0; j < 1; ++j){
            //r_a = *((int4 *)(&A[A_idx]));
            //r_b = *((int4 *)(&B[B_idx]));
            //A_idx += COL_NUM_COUNT;
            //B_idx += COL_NUM_COUNT; 
            // LDG_L1_128bit_LAST(r_a, (const uint8_t*)(&(A[A_idx + i * K_TILE + COL_NUM_COUNT * j])));
            // LDG_L1_128bit_ALWAYS(r_b, (const uint8_t*)(&(B[B_idx + i * K_TILE + COL_NUM_COUNT * j])));
            *(uint4*)(&r_a[0][0]) = *(uint4*)(&A[A_idx + i * K_TILE]);
            *(uint4*)(&r_b[0][0]) = *(uint4*)(&B[B_idx + i * K_TILE]);
            *(uint4*)(&r_a[1][0]) = *(uint4*)(&A[A_idx + i * K_TILE + COL_NUM_COUNT]);
            *(uint4*)(&r_b[1][0]) = *(uint4*)(&B[B_idx + i * K_TILE + COL_NUM_COUNT]);
            //A_idx += COL_NUM_COUNT;
            //B_idx += COL_NUM_COUNT; 

            #pragma unroll 8 
            for(size_t k = 0; k < 8; ++k){
                tmp += r_a[0][k] * r_b[0][k];
            }
        }
        #pragma unroll 8 
        for(size_t k = 0; k < 8; ++k){
            tmp += r_a[1][k] * r_b[1][k];
        }
    }
    #pragma unroll
    for (size_t i = (COL_THREAD >> 1); i > 0;i >>=1 ){
        tmp += __shfl_xor_sync(0xFFFFFFFF, tmp, i, 32);
    }
    if (tid % COL_THREAD == 0){
        size_t C_idx = block_row + warp_row + tid_row;
        C[C_idx] = tmp;
    }
}

//double buff, but N=1
__global__ void mySGemvKernel_Stage(half *__restrict__ A, half *__restrict__ B, half *__restrict__ C, size_t K, size_t N){
    const size_t tid = threadIdx.x % WARP_SIZE; // 0-31
    const size_t warpid = threadIdx.x / WARP_SIZE; // 0-3
    const size_t block_row = blockIdx.x * WARPS_PER_BLOCK * ROW_NUM_COUNT;
    const size_t warp_row = warpid * ROW_NUM_COUNT;
    const size_t tid_row = tid / COL_THREAD;

    const size_t out_K_iters = K / K_TILE; // 32
    // const size_t in_K_iters = K_TILE / COL_NUM_COUNT; // 2
    // const size_t K_iters = K / COL_NUM_COUNT;

    half tmp = 0.0;
    // int4 r_a[2];
    // int4 r_b[2];
    half __align__(16) r_a[2][8];
    half __align__(16) r_b[2][8];

    size_t A_idx = (tid % COL_THREAD) * 8; 
    size_t B_idx = (block_row + warp_row + tid_row) * N + ((tid % COL_THREAD) * 8); 

    // size_t idx = 0;
    // *(uint4*)(&r_a[idx]) = *(uint4*)(&A[A_idx]);
    // *(uint4*)(&r_b[idx]) = *(uint4*)(&B[B_idx]);
    // for (size_t i = 1; i < K_iters; i++) {
    //     #pragma unroll 8 
    //     for(size_t k = 0; k < 8; ++k){
    //         tmp += r_a[idx][k] * r_b[idx][k];
    //     }
    //     idx ^= 1;
    //     *(uint4*)(&r_a[idx]) = *(uint4*)(&A[A_idx + i * COL_NUM_COUNT]);
    //     *(uint4*)(&r_b[idx]) = *(uint4*)(&B[B_idx + i * COL_NUM_COUNT]);
    //     // *(uint4*)(&r_a[idx ^ 1]) = *(uint4*)(&A[A_idx + i * COL_NUM_COUNT]);
    //     // *(uint4*)(&r_b[idx ^ 1]) = *(uint4*)(&B[B_idx + i * K_TILE + COL_NUM_COUNT * (j + 1)]);
    // }
    // #pragma unroll 8 
    // for(size_t k = 0; k < 8; ++k){
    //     tmp += r_a[idx][k] * r_b[idx][k];
    // }
    
    for (size_t i = 0; i < out_K_iters; ++i){
        // size_t idx = 1;
        //r_a = *((int4 *)(&A[A_idx]));
        //r_b = *((int4 *)(&B[B_idx]));
        //A_idx += COL_NUM_COUNT;
        //B_idx += COL_NUM_COUNT; 
        // idx ^= 1;
        // LDG_L1_128bit_LAST(r_a[idx], (const uint8_t*)(&(A[A_idx + i * K_TILE + COL_NUM_COUNT * j])));
        // LDG_L1_128bit_ALWAYS(r_b[idx], (const uint8_t*)(&(B[B_idx + (i * K_TILE + COL_NUM_COUNT * j)])));
        // LDG_L1_128bit_LAST(r_a[idx ^ 1], (const uint8_t*)(&(A[A_idx + i * K_TILE + COL_NUM_COUNT * (j+1) ])));
        // LDG_L1_128bit_ALWAYS(r_b[idx ^ 1], (const uint8_t*)(&(B[B_idx + (i * K_TILE + COL_NUM_COUNT * (j+1))])));
        for (size_t j = 0; j < 1; j++) {
            *(uint4*)(&r_a[0][0]) = *(uint4*)(&A[A_idx + i * K_TILE]);
            *(uint4*)(&r_b[0][0]) = *(uint4*)(&B[B_idx + i * K_TILE]);
            *(uint4*)(&r_a[1][0]) = *(uint4*)(&A[A_idx + i * K_TILE + COL_NUM_COUNT]);
            *(uint4*)(&r_b[1][0]) = *(uint4*)(&B[B_idx + i * K_TILE + COL_NUM_COUNT]);
            // if(threadIdx.x == 0) {
            //     printf("%f, %f \n", __half2float(r_a[0][0]), __half2float(r_b[0][0]));
            // }
            #pragma unroll 8 
            for(size_t k = 0; k < 8; ++k){
                tmp += r_a[0][k] * r_b[0][k];
            }
        }
        #pragma unroll 8 
        for(size_t k = 0; k < 8; ++k){
            tmp += r_a[1][k] * r_b[1][k];
        }
    }
    __syncthreads();
    #pragma unroll
    for (size_t i = (COL_THREAD >> 1); i > 0;i >>=1 ){
        tmp += __shfl_xor_sync(0xFFFFFFFF, tmp, i, 32);
    }
    __syncthreads();
    if (tid % COL_THREAD == 0){
        size_t C_idx = block_row + warp_row + tid_row;
        C[C_idx] = tmp;
    }
}


__global__ void mySGemvKernel_Stage2(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M, size_t N, size_t K){
    const size_t tid = threadIdx.x % WARP_SIZE;
    const size_t warpid = threadIdx.x / WARP_SIZE;
    const size_t block_row = blockIdx.x * WARPS_PER_BLOCK * ROW_NUM_COUNT;
    const size_t warp_row = warpid * ROW_NUM_COUNT;
    const size_t tid_row = tid / COL_THREAD;

    const size_t out_K_iters = K / K_TILE;
    const size_t in_K_iters = K_TILE / COL_NUM_COUNT;

    half tmp = 0.0;
    int4 r_a[in_K_iters];
    int4 r_b[in_K_iters]; 
    size_t B_idx = (tid % COL_THREAD) * 8 + blockIdx.z * K; 
    size_t A_idx = (block_row + warp_row + tid_row) * K + (tid % COL_THREAD) * 8; 

    for (size_t i = 0; i < out_K_iters; ++i){
        #pragma unroll
        for (size_t j = 0; j < (in_K_iters); ++j){
            //r_a = *((int4 *)(&A[A_idx]));
            //r_b = *((int4 *)(&B[B_idx]));
            //A_idx += COL_NUM_COUNT;
            //B_idx += COL_NUM_COUNT; 
            LDG_L1_128bit_LAST(r_a[j], (const uint8_t*)(&(A[A_idx + i * K_TILE + COL_NUM_COUNT * j])));
            LDG_L1_128bit_ALWAYS(r_b[j], (const uint8_t*)(&(B[B_idx + i * K_TILE + COL_NUM_COUNT * j])));
        }
        #pragma unroll
        for(size_t j = 0; j < (in_K_iters); ++j){
            #pragma unroll
            for(size_t k = 0; k < 8; ++k){
                tmp += (reinterpret_cast<half*>(&r_a[j]))[k] * (reinterpret_cast<half*>(&r_b[j]))[k];
            }
        }
    }

    for (size_t i = (COL_THREAD >> 1); i > 0;i >>=1 ){
        tmp += __shfl_xor_sync(0xFFFFFFFF, tmp, i, 32);
    }
    //if (tid % 2 == 0){
    {
        size_t C_idx = block_row + warp_row + tid_row;
        C[C_idx * N + blockIdx.z] = tmp;
    }
}

// double buffer
__global__ void mySGemvKernel_Stage3(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M, size_t N, size_t K){
    const size_t tid = threadIdx.x % WARP_SIZE;
    const size_t warpid = threadIdx.x / WARP_SIZE;
    const size_t block_row = blockIdx.x * WARPS_PER_BLOCK * ROW_NUM_COUNT;
    const size_t warp_row = warpid * ROW_NUM_COUNT;
    const size_t tid_row = tid / COL_THREAD;

    const size_t out_K_iters = K / K_TILE;
    const size_t in_K_iters = K_TILE / COL_NUM_COUNT;

    half tmp = 0.0;
    int4 r_a[2];
    int4 r_b[2]; 
    size_t B_idx = (tid % COL_THREAD) * 8 + blockIdx.z * K; 
    size_t A_idx = (block_row + warp_row + tid_row) * K + (tid % COL_THREAD) * 8; 

    for (size_t i = 0; i < out_K_iters; ++i){
        size_t idx = 0;
        #pragma unroll
        for (size_t j = 0; j < (in_K_iters - 1); ++j){
            //r_a = *((int4 *)(&A[A_idx]));
            //r_b = *((int4 *)(&B[B_idx]));
            //A_idx += COL_NUM_COUNT;
            //B_idx += COL_NUM_COUNT; 
            idx ^= 1;
            LDG_L1_128bit_LAST(r_a[idx], (const uint8_t*)(&(A[A_idx + i * K_TILE + COL_NUM_COUNT * j])));
            LDG_L1_128bit_ALWAYS(r_b[idx], (const uint8_t*)(&(B[B_idx + i * K_TILE + COL_NUM_COUNT * j])));
            LDG_L1_128bit_LAST(r_a[idx ^ 1], (const uint8_t*)(&(A[A_idx + i * K_TILE + COL_NUM_COUNT * (j+1) ])));
            LDG_L1_128bit_ALWAYS(r_b[idx ^ 1], (const uint8_t*)(&(B[B_idx + i * K_TILE + COL_NUM_COUNT * (j+1) ])));

            #pragma unroll 
            for(size_t k = 0; k < 8; ++k){
                tmp += (reinterpret_cast<half*>(&r_a[idx]))[k] * (reinterpret_cast<half*>(&r_b[idx]))[k];
            }
        }
            #pragma unroll 
            for(size_t k = 0; k < 8; ++k){
                tmp += (reinterpret_cast<half*>(&r_a[idx ^ 1]))[k] * (reinterpret_cast<half*>(&r_b[idx ^ 1]))[k];
            }
    }
    #pragma unroll
    for (size_t i = (COL_THREAD >> 1); i > 0;i >>=1 ){
        tmp += __shfl_xor_sync(0xFFFFFFFF, tmp, i, 32);
    }
    //if (tid % 2 == 0){
    {
        size_t C_idx = block_row + warp_row + tid_row;
        C[C_idx * N + blockIdx.z] = tmp;
    }
}

#define begin_N 8 
// vectorize B
__global__ void mySGemvKernel_Stage4(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M, size_t N, size_t K){
    const size_t tid = threadIdx.x % WARP_SIZE;
    const size_t warpid = threadIdx.x / WARP_SIZE;
    const size_t block_row = blockIdx.x * WARPS_PER_BLOCK * ROW_NUM_COUNT;
    const size_t warp_row = warpid * ROW_NUM_COUNT;
    const size_t tid_row = tid / COL_THREAD;

    const size_t out_K_iters = K / K_TILE;
    constexpr size_t in_K_iters = K_TILE / COL_NUM_COUNT;

    half tmp[begin_N] = {0.0};
    int4 r_a[in_K_iters];
    int4 r_b[in_K_iters][begin_N]; 
    size_t B_idx = (tid % COL_THREAD) * 8 + blockIdx.z * K; 
    size_t A_idx = (block_row + warp_row + tid_row) * K + (tid % COL_THREAD) * 8; 

    for (size_t i = 0; i < out_K_iters; ++i){
        //#pragma unroll
        for (size_t j = 0; j < (in_K_iters); ++j){
            LDG_L1_128bit_LAST(r_a[j], (const uint8_t*)(&(A[A_idx + i * K_TILE + COL_NUM_COUNT * j])));
            for(size_t n = 0; n < begin_N; ++n){
                LDG_L1_128bit_ALWAYS(r_b[j][n], (const uint8_t*)(&(B[B_idx + i * K_TILE + COL_NUM_COUNT * j + K * n])));
            }
        }

        for(size_t j = 0; j < (in_K_iters); ++j){
            for(size_t n = 0; n < begin_N; ++n){
                //#pragma unroll 
                for(size_t k = 0; k < 8; ++k){
                    tmp[n] += (reinterpret_cast<half*>(&r_a[j]))[k] * (reinterpret_cast<half*>(&r_b[j][n]))[k];
                }
            }
        }
    }
    //printf("tid  : %d, %.4f, %.4f\n", threadIdx.x, float(tmp[0]), float(tmp[1]));

    for (size_t i = (COL_THREAD >> 1); i > 0;i >>=1 ){
        for(size_t n = 0; n < begin_N; ++n){
            tmp[n] += __shfl_xor_sync(0xFFFFFFFF, tmp[n], i, 32);}
    }
    //printf("tid  : %d, %.4f, %.4f\n", threadIdx.x, float(tmp[0]), float(tmp[1]));
    //if (tid % 2 == 0){
    size_t C_idx = block_row + warp_row + tid_row;
    //#pragma unroll
    for(size_t n = 0; n < begin_N; ++n)
    {
        //*(int4*)(&C[C_idx * N + blockIdx.z * 8]) = *(int4*)(&tmp[0]);
        C[C_idx * N + n + blockIdx.z * 8] = tmp[n];
    }
}


// vectorize B + double buff
__global__ void mySGemvKernel_Stage5(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M, size_t N, size_t K){
    const size_t tid = threadIdx.x % WARP_SIZE;
    const size_t warpid = threadIdx.x / WARP_SIZE;
    const size_t block_row = blockIdx.x * WARPS_PER_BLOCK * ROW_NUM_COUNT;
    const size_t warp_row = warpid * ROW_NUM_COUNT;
    const size_t tid_row = tid / COL_THREAD;

    const size_t out_K_iters = K / K_TILE;
    const size_t in_K_iters = K_TILE / COL_NUM_COUNT;

    half tmp[8] = {0.0f};
    int4 r_a[2];
    int4 r_b[2]; 
    size_t B_idx = (tid % COL_THREAD) * 8 + blockIdx.z * K; 
    size_t A_idx = (block_row + warp_row + tid_row) * K + (tid % COL_THREAD) * 8; 

    for (size_t i = 0; i < out_K_iters; ++i){
        size_t idx = 1;
        #pragma unroll
        for (size_t j = 0; j < (in_K_iters - 1); ++j){
            idx ^= 1;
            //r_a = *((int4 *)(&A[A_idx]));
            //r_b = *((int4 *)(&B[B_idx]));
            //A_idx += COL_NUM_COUNT;
            //B_idx += COL_NUM_COUNT; 
            LDG_L1_128bit_LAST(r_a[idx], (const uint8_t*)(&(A[A_idx + i * K_TILE + COL_NUM_COUNT * j])));
            LDG_L1_128bit_LAST(r_a[idx ^ 1], (const uint8_t*)(&(A[A_idx + i * K_TILE + COL_NUM_COUNT * (j+1)])));
            LDG_L1_128bit_ALWAYS(r_b[idx], (const uint8_t*)(&(B[B_idx + i * K_TILE + COL_NUM_COUNT * j])));
            LDG_L1_128bit_ALWAYS(r_b[idx ^ 1], (const uint8_t*)(&(B[B_idx + i * K_TILE + COL_NUM_COUNT * (j + 1)])));
       
            for(size_t n = 0; n < N; ++n){
                LDG_L1_128bit_ALWAYS(r_b[idx], (const uint8_t*)(&(B[B_idx + i * K_TILE + COL_NUM_COUNT * j + K * (n)])));
                #pragma unroll 
                for(size_t k = 0; k < 8; ++k){
                    tmp[n] += (reinterpret_cast<half*>(&r_a[idx]))[k] * (reinterpret_cast<half*>(&r_b[idx]))[k];
                }
            }
        }
        for(size_t n = 0; n < N; ++n){
            LDG_L1_128bit_ALWAYS(r_b[idx ^ 1], (const uint8_t*)(&(B[B_idx + i * K_TILE + COL_NUM_COUNT * (in_K_iters - 1) + K * n])));
            #pragma unroll 
            for(size_t k=0;k < 8;++k){
                tmp[n] += (reinterpret_cast<half*>(&r_a[idx ^ 1]))[k] * (reinterpret_cast<half*>(&r_b[idx ^ 1]))[k];
            }
        }
    }
    //printf("tid  : %d, %.4f, %.4f\n", threadIdx.x, float(tmp[0]), float(tmp[1]));

    for (size_t i = (COL_THREAD >> 1); i > 0;i >>=1 ){
        for(size_t n = 0; n < N; ++n){
            tmp[n] += __shfl_xor_sync(0xFFFFFFFF, tmp[n], i, 32);}
    }
    //printf("tid  : %d, %.4f, %.4f\n", threadIdx.x, float(tmp[0]), float(tmp[1]));
    //if (tid % 2 == 0){
    size_t C_idx = block_row + warp_row + tid_row;
    //*(int4*)(&C[C_idx * N + blockIdx.z * 8]) = *(int4*)(&tmp[0]);
    //#pragma unroll
    for(size_t n = 0; n < N; ++n)
    {
        C[C_idx * N + n + blockIdx.z * 8] = tmp[n];
    }
}

//__global__ void warp2NaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t N,
//                                 size_t K) {
//    const size_t group_id = threadIdx.x / GROUP_SIZE;
//    const size_t group_col = blockIdx.x * COLS_PER_BLOCK + group_id;
//    if (group_col >= N) {
//        return;
//    }
//
//    const size_t K_iters = div_ceil(K, GROUP_SIZE);
//    const size_t group_lane_id = threadIdx.x % GROUP_SIZE;
//
//    float tmp = 0.0;
//#pragma unroll
//    for (size_t i = 0; i < K_iters; ++i) {
//        size_t A_idx = i * GROUP_SIZE + group_lane_id;
//        size_t B_idx = i * GROUP_SIZE + group_lane_id + group_col * K;
//        tmp += __half2float(A[A_idx]) * __half2float(B[B_idx]);
//    }
//
//    const unsigned int mask = 0xffffffff;
//#pragma unroll
//    for (size_t i = GROUP_SIZE / 2; i >= 1; i /= 2) {
//        tmp += __shfl_xor_sync(mask, tmp, i);
//    }
//
//    if (group_lane_id == 0) {
//        C[group_col] = __float2half(tmp);
//    }
//}
//
//void warp2Naive(half *A, half *B, half *C, size_t N, size_t K) {
//    dim3 block(THREADS_PER_BLOCK);
//    dim3 grid(div_ceil(N, COLS_PER_BLOCK));
//
//    warp2NaiveKernel<<<grid, block>>>(A, B, C, N, K);
//}

template<F16F16GemmTCAlgo_t algo = HGEMMAlignedV1>
void myF16F16GemmTCWarp(half *a, half *b, half *c, int M, int N, int K) {

    if (algo == HGEMMAlignedV1) {
        //warp16Smem(a, b, c, N, K);
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(div_ceil(N, BLOCK_ROW));
        mySGemvKernel<<<grid, block>>>(a, b, c, K, N);
    }
    else if(algo == HGEMMAlignedV2){
        //warp2Naive(a, b, c, N, K);
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(div_ceil(N, BLOCK_ROW));
        mySGemvKernel_Stage<<<grid, block>>>(a, b, c, K, N);
    }
    else if(algo == HGEMMAlignedV3){
        //warp2Naive(a, b, c, N, K);
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(M / BLOCK_ROW, 1, N);
        mySGemvKernel_Stage2<<<grid, block>>>(a, b, c, M, N, K);
    }else if(algo == HGEMMAlignedV4){
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(M / BLOCK_ROW, 1, N);
        mySGemvKernel_Stage3<<<grid, block>>>(a, b, c, M, N, K);
    }else if(algo == HGEMMAlignedV5){
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(M / BLOCK_ROW, 1, (N + 1) / begin_N);
        mySGemvKernel_Stage4<<<grid, block>>>(a, b, c, M, N, K);     
    }else if(algo == HGEMMAlignedV6){
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(M / BLOCK_ROW, 1, (N + 8 - 1) / 8);
        mySGemvKernel_Stage5<<<grid, block>>>(a, b, c, M, N, K);
    }
}
float testF16F16GemmMaxError(
    void (*gpuF16F16Gemm) (half *, half *, half *, int, int, int),
    int M, int N, int K) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *h_a, *h_b, *d_a, *d_b;
    half *h_c, *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_c = (half *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (half *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++){
        h_a[i] = (half)(float(rand()) / float(RAND_MAX));
        //h_a[i] = (half)(float(i) / 100);
    }
    for (int i = 0; i < K * N; i++)
        //h_b[i] = (half)(rand() / float(RAND_MAX));
        h_b[i] = (half)(float(i)/ 10000);

    cpuF16F16Gemm(h_a, h_b, h_c, M, N, K);

    for (int i =0; i < N; ++i){
        for(int j = 0; j < K; ++j){
           h_b[i * K + j] = (half)((float)(j * N + i) / 10000);
        }
    }

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            printf("%f , ", float(h_c[i * N + j]));
        }
        printf("\n----------------------\n");
    }
    printf("\n\n\n\n\n\n\n");
    for(int i = 0; i < N; ++i){ 
        for(int j = 0; j < M; ++j){
            printf("%f , ", float(h_d_c[i * N + j]));
        }
        printf("\n======================\n");
    }
    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs((float)h_d_c[i] - (float)h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); free(h_d_c);

    return max_error;
}
float testF16F16GemmPerformance(
    void (*gpuF16F16Gemm) (half *, half *, half *, int, int, int),
    int M, int N, int K, int repeat) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *d_a, *d_b;
    half *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    
    half *h_a, *h_b;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);

    for (int i = 0; i < M * K; i++){
        h_a[i] = (half)(1.0);
    }
    for (int i = 0; i < K * N; i++)
        h_b[i] = (half)(1.0);
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    //warmup 
    for(int i = 0; i < 10;i++){
        gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gpuF16F16Gemm(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec = 0.f;
    cudaEventElapsedTime(&msec, start, end);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    free(h_a); free(h_b);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return double(msec / repeat);
}
int main(int arg, char* argv[]){
    // printf("\nalgo = HGEMMAlignedV1\n");
    const int test_num = 1;
    //const int test_num = 16;
    //// [1, 4096, 1024],  [1, 2304, 768], [1, 768, 3072]
    //const int M_list[test_num] = {8192, 50, 50, 500, 256,  1024, 768, 1024, 4096, 2304, 768, 3072, 4096, 4096, 11008};
    ////const int N_list[test_num] = {1,   1,   1,   1,    1   , 1,    1,   1,    1,    1    , 1};
    //const int N_list[test_num] = {5,   5,   5,   5,    5   , 5,    5,   5, 5, 5, 5};
    ////const int N_list[test_num] = {8,   8,   8,   8,    8   , 8,    8,   8,    8,    8,   8};
    //const int K_list[test_num] = {14208, 64, 64, 128, 128, 256, 1024, 768, 4096, 1024, 768, 3072, 768, 4096, 11008, 4096};
    const int outer_repeat = 1, inner_repeat = 10;
    //const int total_N = 5;
    //{
    //    const int M = 512, N = 19, K = 512;
    //    //const int M = 16, N = 2, K = 32;
    //    float max_error = testF16F16GemmMaxError(
    //        myF16F16GemmTCWarp<HGEMMAlignedV5>, M, N, K);
    //    printf("Max Error = %f\n", max_error);
    //}

    //printf("-------------------------block z----------------------------\n");
    //for (int j = 0; j < test_num; j++){
    //    int M = std::atoi(argv[2]), N = std::atoi(argv[1]), K = std::atoi(argv[3]);
    //    //int M = 1024 * j, N = 1, K = 1024 * j;

    //    double max_sec = 0.0;
    //    double min_sec = DBL_MAX;
    //    double total_sec = 0.0;

    //    for (int k = 0; k < outer_repeat; k++) {
    //        double this_sec = testF16F16GemmPerformance(
    //            myF16F16GemmTCWarp<HGEMMAlignedV3>, M, N, K, inner_repeat);
    //        max_sec = max(max_sec, this_sec);
    //        min_sec = min(min_sec, this_sec);
    //        total_sec += this_sec;
    //    }

    //    double avg_sec = total_sec / outer_repeat;
    //    double avg_Gflops = ((long)M) * N * K * 2 / 1e12 / (avg_sec / 1e3);

    //    printf("M N K = %6d %6d %6d, ", M, N, K);
    //    printf("Time = %12.8lf %12.8lf %12.8lf ms, ", min_sec, avg_sec, max_sec);
    //    printf("AVG Performance = %10.4lf Tflops\n", avg_Gflops);
    //}

    //printf("---------------------blockz--double-buffer-----------------\n");
    //for (int j = 0; j < test_num; j++){
    //    //int M = M_list[j], N = total_N, K = K_list[j];
    //    int M = std::atoi(argv[2]), N = std::atoi(argv[1]), K = std::atoi(argv[3]);
    //    //int M = 1024 * j, N = 1, K = 1024 * j;

    //    double max_sec = 0.0;
    //    double min_sec = DBL_MAX;
    //    double total_sec = 0.0;

    //    for (int k = 0; k < outer_repeat; k++) {
    //        double this_sec = testF16F16GemmPerformance(
    //            myF16F16GemmTCWarp<HGEMMAlignedV4>, M, N, K, inner_repeat);
    //        max_sec = max(max_sec, this_sec);
    //        min_sec = min(min_sec, this_sec);
    //        total_sec += this_sec;
    //    }

    //    double avg_sec = total_sec / outer_repeat;
    //    double avg_Gflops = ((double)M) * N * K * 2 / 1e12 / (avg_sec / 1e3);

    //    printf("M N K = %6d %6d %6d, ", M, N, K);
    //    printf("Time = %12.8lf %12.8lf %12.8lf ms, ", min_sec, avg_sec, max_sec);
    //    printf("AVG Performance = %10.4lf Tflops\n", avg_Gflops);
    //}
    // printf("----------------cache A for B------------------------\n");
    for (int j = 0; j < test_num; j++){
        //int M = M_list[j], N = total_N, K = K_list[j];
        int M = std::atoi(argv[1]), N = std::atoi(argv[2]), K = std::atoi(argv[3]);
        //int M = 1024 * j, N = 1, K = 1024 * j;

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = testF16F16GemmPerformance(
                myF16F16GemmTCWarp<HGEMMAlignedV1>, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1e12 / (avg_sec / 1e3);

        printf("M N K = %6d %6d %6d, ", M, N, K);
        printf("Time = %12.8lf %12.8lf %12.8lf us, ", min_sec * 1e3, avg_sec * 1e3, max_sec * 1e3);
        printf("AVG Performance = %10.4lf Gflops\n", avg_Gflops);
    }
    ////printf("-------------------cache A for B double-buff---------------------\n");
    //for (int j = 0; j < test_num; j++){
    //    //int M = M_list[j], N = total_N, K = K_list[j];
    //    int M = std::atoi(argv[2]), N = std::atoi(argv[1]), K = std::atoi(argv[3]);
    //    //int M = 1024 * j, N = 1, K = 1024 * j;

    //    double max_sec = 0.0;
    //    double min_sec = DBL_MAX;
    //    double total_sec = 0.0;

    //    for (int k = 0; k < outer_repeat; k++) {
    //        double this_sec = testF16F16GemmPerformance(
    //            myF16F16GemmTCWarp<HGEMMAlignedV6>, M, N, K, inner_repeat);
    //        max_sec = max(max_sec, this_sec);
    //        min_sec = min(min_sec, this_sec);
    //        total_sec += this_sec;
    //    }

    //    double avg_sec = total_sec / outer_repeat;
    //    double avg_Gflops = ((double)M) * N * K * 2 / 1e12 / (avg_sec / 1e3);

    //    printf("M N K = %6d %6d %6d, ", M, N, K);
    //    printf("Time = %12.8lf %12.8lf %12.8lf ms, ", min_sec, avg_sec, max_sec);
    //    printf("AVG Performance = %10.4lf Tflops\n", avg_Gflops);
    //}
    return 0;
}

