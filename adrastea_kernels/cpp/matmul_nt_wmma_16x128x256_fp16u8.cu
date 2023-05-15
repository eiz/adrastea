#include "compat.h"

#include <mma.h>

__device__ __forceinline__ half dequantize_absmax_one(uint8_t x, half scale) {
  return (float(x) - 127.5f) * __half2float(scale);
}

// output = lhs * rhs^T; lhs = (m, p); rhs = (n, p); output = (m, n)
// blockDim must be (256, 1, 1). gridDim must be (m/16 * n/128, 1, 1)
// m must be a multiple of 16, n must be a multiple of 128 and p a multiple of 256
extern "C" __global__ void matmul_nt_wmma_16x128x256_fp16u8(half* __restrict__ output,
                                                            half const* __restrict__ lhs,
                                                            uint8_t const* __restrict__ rhs,
                                                            half const* __restrict__ rhs_scale,
                                                            int m,
                                                            int p,
                                                            int n,
                                                            int block_size,
                                                            float beta = 0.0f) {
  using namespace nvcuda::wmma;
  extern __shared__ void* sdata[];
  const int SDATA_BASE_LHS = 0;
  const int SDATA_BASE_RHS = 16 * 272;
#define SDATA(type, side, stride, d0, d1) \
  (((type*)sdata)[SDATA_BASE_##side + ((d0) * (stride)) + (d1)])
#define LHS(d0, d1) SDATA(__half, LHS, 272, d0, d1)
#define RHS(d0, d1) SDATA(__half, RHS, 272, d0, d1)
#define OUT(d0, d1) SDATA(float, LHS, 128, d0, d1)
  int bid = BLOCK_IDX_X;
  int dim_y = m / 16;
  int bx = (bid / dim_y) * 128, by = (bid % dim_y) * 16;
  unsigned tid = THREAD_IDX_X;
  int tlo = tid & 63, thi = tid >> 6;
  int warp_id = tid / 32;
  int wx = 32 * (warp_id >> 1);
  int block_count = p / block_size;
  fragment<accumulator, 16, 16, 16, float> frag_accum[2];
  fragment<matrix_a, 16, 16, 16, __half, row_major> frag_lhs;
  fragment<matrix_b, 16, 16, 16, __half, col_major> frag_rhs[2];
  fill_fragment(frag_accum[0], 0.0f);
  fill_fragment(frag_accum[1], 0.0f);
  for (int t = 0; t < p; t += 256) {
    for (int i = 0; i < 4; ++i) {
      int lhs_idx = (by + thi * 4 + i) * p + t + tlo * 4;
      *((short4*)&LHS(thi * 4 + i, tlo * 4)) = *(short4*)(&lhs[lhs_idx]);
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      half scale =
          __ldg(&rhs_scale[(bx + thi * 32 + i) * block_count + (t + tlo * 4) / block_size]);
      int rhs_idx = (bx + thi * 32 + i) * p + t + tlo * 4;
      uint32_t rhs_unscaled = *((uint32_t*)&rhs[rhs_idx]);
      half rhs_scaled[4];
      for (int j = 0; j < 4; ++j) {
        rhs_scaled[j] = dequantize_absmax_one((rhs_unscaled >> (8 * j)) & 0xFF, scale);
      }
      *((short4*)&RHS(thi * 32 + i, tlo * 4)) = *(short4*)rhs_scaled;
    }
    __syncthreads();
    for (int i = 0; i < 256; i += 16) {
      load_matrix_sync(frag_lhs, &LHS(0, i), 272);
      load_matrix_sync(frag_rhs[0], &RHS(wx, i), 272);
      mma_sync(frag_accum[0], frag_lhs, frag_rhs[0], frag_accum[0]);
      load_matrix_sync(frag_rhs[1], &RHS(wx + 16, i), 272);
      mma_sync(frag_accum[1], frag_lhs, frag_rhs[1], frag_accum[1]);
    }
    __syncthreads();
  }
  store_matrix_sync(&OUT(0, wx), frag_accum[0], 128, mem_row_major);
  store_matrix_sync(&OUT(0, wx + 16), frag_accum[1], 128, mem_row_major);
  __syncthreads();
  int tx = (tid & 15) * 8, ty = (tid >> 4);
  int r = by + ty, c = bx + tx;
  for (int j = 0; j < 8; ++j) {
    int out_idx = r * n + c + j;
    output[out_idx] = OUT(ty, tx + j) + beta * __half2float(output[out_idx]);
  }
#undef SDATA
#undef LHS
#undef RHS
#undef OUT
}
