#include "compat.h"

// TODO broken kernel

// output = lhs * rhs^T; lhs = (m, p); rhs = (n, p); output = (m, n)
// blockDim must be (256, 1, 1). gridDim must be (m/128 * n/64, 1, 1)
// m must be a multiple of 128, n and p must be multiples of 64
extern "C" __global__ void matmul_nt_128x64x64(__half* __restrict__ output,
                                               __half const* __restrict__ lhs,
                                               __half const* __restrict__ rhs,
                                               int m,
                                               int p,
                                               int n,
                                               float beta = 0.0f) {
  extern __shared__ void* sdata[];
  const int SDATA_BASE_LHS = 0;
  const int SDATA_BASE_RHS = 128 * 80;
#define SDATA(type, side, stride, d0, d1) \
  (((type*)sdata)[SDATA_BASE_##side + ((d0) * (stride)) + (d1)])
#define LHS(d0, d1) SDATA(__half, LHS, 80, d0, d1)
#define RHS(d0, d1) SDATA(__half, RHS, 80, d0, d1)
  int bid = BLOCK_IDX_X;
  int dim_y = m / 128;
  int bx = (bid / dim_y) * 64, by = (bid % dim_y) * 128;
  unsigned tid = THREAD_IDX_X;
  float sum[32];
  for (int t = 0; t < p; t += 64) {
    {
      int tlo = tid & 15, thi = tid >> 4;  // 16x16 grid
      for (int i = 0; i < 8; ++i) {
        int lhs_idx = (by + thi * 8 + i) * p + t + tlo * 4;
        *((short4*)&LHS(thi * 8 + i, tlo * 4)) = *(short4*)(&lhs[lhs_idx]);
      }
      for (int i = 0; i < 4; ++i) {
        int rhs_idx = (bx + thi * 4 + i) * p + t + tlo * 4;
        *((short4*)&RHS(thi * 4 + i, tlo * 4)) = *(short4*)(&rhs[rhs_idx]);
      }
    }
    __syncthreads();
    {
      int tlo = tid & 3, thi = tid >> 2;  // 64x4 grid
      for (int pp = 0; pp < 64; ++pp) {
        for (int mm = 0; mm < 2; ++mm) {
          for (int nn = 0; nn < 16; ++nn) {
            sum[mm * 16 + nn] +=
                __half2float(LHS(thi * 2 + mm, pp)) * __half2float(RHS(tlo * 16 + nn, pp));
          }
        }
      }
    }
    __syncthreads();
  }
  __syncthreads();
  {
    int tlo = tid & 3, thi = tid >> 2;  // 64x4 grid
    int tx = tlo * 16, ty = thi * 2;
    int r = by + ty, c = bx + tx;
    for (int mm = 0; mm < 2; ++mm) {
      for (int nn = 0; nn < 16; ++nn) {
        int out_idx = (r + mm) * n + c + nn;
        output[out_idx] = sum[mm * 16 + nn] + beta * __half2float(output[out_idx]);
      }
    }
  }
#undef SDATA
#undef LHS
#undef RHS
#undef OUT
}
