#include <cuda_fp16.h>
#include <mma.h>

// output = lhs * rhs^T; lhs = (m, p); rhs = (n, p); output = (m, n)
// blockDim must be (256, 1, 1). gridDim must be (m/128 * n/64, 1, 1)
// m must be a multiple of 128, n and p must be multiples of 64
extern "C" __global__ void matmul_nt_wmma_128x64x64(__half* __restrict__ output,
                                                    __half const* __restrict__ lhs,
                                                    __half const* __restrict__ rhs,
                                                    int m,
                                                    int p,
                                                    int n,
                                                    float beta = 0.0f) {
  using namespace nvcuda::wmma;
  extern __shared__ void* sdata[];
  const int SDATA_BASE_LHS = 0;
  const int SDATA_BASE_RHS = 128 * 80;
#define SDATA(type, side, stride, d0, d1) \
  (((type*)sdata)[SDATA_BASE_##side + ((d0) * (stride)) + (d1)])
#define LHS(d0, d1) SDATA(__half, LHS, 80, d0, d1)
#define RHS(d0, d1) SDATA(__half, RHS, 80, d0, d1)
#define OUT(d0, d1) SDATA(float, LHS, 64, d0, d1)
  int bid = blockIdx.x;
  int dim_y = m / 128;
  int bx = (bid / dim_y) * 64, by = (bid % dim_y) * 128;
  unsigned tid = threadIdx.x;
  int tlo = tid & 15, thi = tid >> 4;
  int warp_id = tid / 32;
  int wx = 32 * (warp_id & 1);
  int wy = 32 * (warp_id >> 1);
  fragment<accumulator, 16, 16, 16, float> frag_accum[2][2];
  fragment<matrix_a, 16, 16, 16, __half, row_major> frag_lhs[2];
  fragment<matrix_b, 16, 16, 16, __half, col_major> frag_rhs[2];
  fill_fragment(frag_accum[0][0], 0.0f);
  fill_fragment(frag_accum[0][1], 0.0f);
  fill_fragment(frag_accum[1][0], 0.0f);
  fill_fragment(frag_accum[1][1], 0.0f);
  for (int t = 0; t < p; t += 64) {
    for (int i = 0; i < 8; ++i) {
      int lhs_idx = (by + thi * 8 + i) * p + t + tlo * 4;
      *((short4*)&LHS(thi * 8 + i, tlo * 4)) = *(short4*)(&lhs[lhs_idx]);
    }
    for (int i = 0; i < 4; ++i) {
      int rhs_idx = (bx + thi * 4 + i) * p + t + tlo * 4;
      *((short4*)&RHS(thi * 4 + i, tlo * 4)) = *(short4*)(&rhs[rhs_idx]);
    }
    __syncthreads();
    for (int i = 0; i < 64; i += 16) {
      load_matrix_sync(frag_lhs[0], &LHS(wy, i), 80);
      load_matrix_sync(frag_rhs[0], &RHS(wx, i), 80);
      mma_sync(frag_accum[0][0], frag_lhs[0], frag_rhs[0], frag_accum[0][0]);
      load_matrix_sync(frag_rhs[1], &RHS(wx + 16, i), 80);
      mma_sync(frag_accum[0][1], frag_lhs[0], frag_rhs[1], frag_accum[0][1]);
      load_matrix_sync(frag_lhs[1], &LHS(wy + 16, i), 80);
      mma_sync(frag_accum[1][0], frag_lhs[1], frag_rhs[0], frag_accum[1][0]);
      mma_sync(frag_accum[1][1], frag_lhs[1], frag_rhs[1], frag_accum[1][1]);
    }
    __syncthreads();
  }
  store_matrix_sync(&OUT(wy, wx), frag_accum[0][0], 64, mem_row_major);
  store_matrix_sync(&OUT(wy, wx + 16), frag_accum[0][1], 64, mem_row_major);
  store_matrix_sync(&OUT(wy + 16, wx), frag_accum[1][0], 64, mem_row_major);
  store_matrix_sync(&OUT(wy + 16, wx + 16), frag_accum[1][1], 64, mem_row_major);
  __syncthreads();
  int tx = tlo * 4, ty = thi * 8;
  int r = by + ty, c = bx + tx;
  for (int k = 0; k < 8; ++k) {
    for (int j = 0; j < 4; ++j) {
      int out_idx = (r + k) * n + c + j;
      output[out_idx] = OUT(ty + k, tx + j) + beta * __half2float(output[out_idx]);
    }
  }
#undef SDATA
#undef LHS
#undef RHS
#undef OUT
}
