#include <cuda_fp16.h>
#include <cuda.h> 

__device__ __forceinline__ void storeSmemC(half *C, float *smem, int M, int N) {
  // load 128 * 128
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int tid = tz * 64 + ty * 32 + tx;
  for (int i = 0; i < 128; ++i) {
    int row = i;
    int col = tid;
    int scol = col ^ ((row & 3) << 3);
    (C[(by * 128 + row) * N + bx * 128 + col]) = (half)smem[row * 128 + scol];
  }
}

__device__ __forceinline__ void loadFragA(unsigned int *frag, half *smem,
                                          int ki) {
  // frag: [j, k]: [2, 2]
  // load 64x16
  int tx = threadIdx.x;
  int tz = threadIdx.z;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        int row = tz * 64 + i * 16 + j * 8 + tx / 4;
        int col = ki * KII + k * 8 + tx % 4 * 2;
        col = row % 2 * 32 + col;
        row = row / 2;
        col = col ^ ((row & 3) << 3);
        unsigned int *ptr =
            reinterpret_cast<unsigned int *>(smem + row * 64 + col);
        frag[i * 4 + j * 2 + k] = ptr[0];
      }
    }
  }

//   load 16x16 at a time
//   #pragma unroll
//     for (int i = 0; i < 4; ++i) {
//       int row = tz * 64 + i * 16 + tx / 16 * 8 + tx % 8;
//       int col = ki * KII + tx / 8 % 2 * 8;
//       col = row % 2 * 32 + col;
//       row = row / 2;
//       col = col ^ (((row & 3) << 3));
//       void *ptr = (void *)(smem + row * 64 + col);
//       uint32_t smem_ptr;
//       asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64"
//           "%0, smem_ptr; }\n"
//           : "=r"(smem_ptr)
//           : "l"(ptr));
//       asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" 
//           : "=r"(frag[i * 4 + 0]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]), "=r"(frag[i * 4 + 3])
//           : "r"(smem_ptr));
//     }
}

__device__ __forceinline__ void loadFragB(unsigned int *frag, half *smem,
                                          int ki) {
  // frag: [j, k]: []
  // load 64x16
  int tx = threadIdx.x;
  int ty = threadIdx.y;

//   for (int i = 0; i < 4; ++i) {
//     for (int j = 0; j < 2; ++j) {
//       for (int k = 0; k < 2; ++k) {
//         int row = ty * 64 + i * 16 + j * 8 + tx / 4;
//         int col = ki * KII + k * 8 + tx % 4 * 2;
//         col = row % 2 * 32 + col;
//         row = row / 2;
//         col = col ^ ((row & 3) << 3);
//         unsigned int *ptr =
//             reinterpret_cast<unsigned int *>(smem + row * 64 + col);
//         frag[i * 4 + j * 2 + k] = ptr[0];
//       }
//     }
//   }

// load 16x16 at a time
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = ty * 64 + i * 16 + tx / 16 * 8 + tx % 8;
    int col = ki * KII + tx / 8 % 2 * 8;
    col = row % 2 * 32 + col;
    row = row / 2;
    col = col ^ (((row & 3) << 3));
    void *ptr = (void *)(smem + row * 64 + col);
    uint32_t smem_ptr;
    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag[i * 4 + 0]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]),
          "=r"(frag[i * 4 + 3])
        : "r"(smem_ptr));
  }
}

__device__ __forceinline__ void storeAccum(float *ptr, float *frag) {
  // frag [r, c, _]: [2, 2, 2]
  // store 64x64
  //   int tx = threadIdx.x;
  //   int ty = threadIdx.y;
  //   int tz = threadIdx.z;
  //   int row = tz * 64 + tx / 4;
  //   int col = ty * 64 + tx % 4 * 2;
  //   // float *dst = ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) +
  //   row %
  //   // 16 * 16 + col % 16;
  //   float *dst = ptr + row * 128 + col;
  // #pragma unroll
  //   for (int i = 0; i < 4; ++i) {
  // #pragma unroll
  //     for (int j = 0; j < 4; ++j) {
  //       dst[0] = frag[i * 32 + j * 8 + 0 * 4 + 0 * 2];
  //       dst[1] = frag[i * 32 + j * 8 + 0 * 4 + 0 * 2 + 1];

  //       dst[0 + 8] = frag[i * 32 + j * 8 + 0 * 4 + 1 * 2];
  //       dst[1 + 8] = frag[i * 32 + j * 8 + 0 * 4 + 1 * 2 + 1];

  //       dst[0 + 8 * 128] = frag[i * 32 + j * 8 + 1 * 4 + 0 * 2];
  //       dst[1 + 8 * 128] = frag[i * 32 + j * 8 + 1 * 4 + 0 * 2 + 1];

  //       dst[0 + 8 * 128 + 8] = frag[i * 32 + j * 8 + 1 * 4 + 1 * 2];
  //       dst[1 + 8 * 128 + 8] = frag[i * 32 + j * 8 + 1 * 4 + 1 * 2 + 1];

  //       dst += 16;
  //     }
  //     dst += 16 * 128 - 16 * 4;
  //   }
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  // smem view is [128x128]
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
          int row = tz * 64 + i * 16 + r * 8 + tx / 4;
          int col = ty * 64 + j * 16 + c * 8 + tx % 4 * 2;
          col = col ^ ((row & 3) << 3);
          //   float2* dst = reinterpret_cast<float2*>(ptr + row * 128 + col);
          //   float2 tmp;
          //   tmp.x = frag[i * 32 + j * 8 + r * 4 + c * 2 + 0];
          //   tmp.y = frag[i * 32 + j * 8 + r * 4 + c * 2 + 1];
          //   *dst = tmp;
          ptr[row * 128 + col] = frag[i * 32 + j * 8 + r * 4 + c * 2 + 0];
          ptr[row * 128 + (col + 1)] = frag[i * 32 + j * 8 + r * 4 + c * 2 + 1];
        }
      }
    }
  }
}

__device__ __forceinline__ void mma_sync(float* accum, uint32_t* fragA, uint32_t* fragB) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10, %11, %12, %13};\n"
                : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
                : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
                  "r"(fragB[0]), "r"(fragB[1]), "f"(accum[0]), "f"(accum[1]),
                  "f"(accum[4]), "f"(accum[5]));

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5,  %6,  %7},"
                "{%8,  %9},"
                "{%10, %11, %12, %13};\n"
                : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
                : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
                  "r"(fragB[2]), "r"(fragB[3]), "f"(accum[2]), "f"(accum[3]),
                  "f"(accum[6]), "f"(accum[7]));
}