#include "compat.h"

// output = lhs * rhs^T + bias + output * beta; lhs = (m, k); rhs = (n, k); output = (m, n)
// this only exists as a trivial reference point for optimized kernels.
extern "C" __global__ void linear(__half* const output,
                                  __half const* const lhs,
                                  __half const* const rhs,
                                  __half const* const bias,
                                  int const m,
                                  int const k,
                                  int const n,
                                  int const stride_ox,
                                  int const stride_oy,
                                  int const stride_lx,
                                  int const stride_ly,
                                  int const stride_rx,
                                  int const stride_ry,
                                  float const beta = 0.0f) {
  int c = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int r = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  if (r < m && c < n) {
    float sum = 0;
    for (int i = 0; i < k; i++) {
      sum += __half2float(lhs[r * stride_ly + i * stride_lx]) *
             __half2float(rhs[c * stride_ry + i * stride_rx]);
    }
    output[r * stride_oy + c * stride_ox] =
        sum + __half2float(bias[c]) + beta * __half2float(output[r * stride_oy + c * stride_ox]);
  }
}
