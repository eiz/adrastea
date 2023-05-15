#include "compat.h"

extern "C" __global__ void silu(__half* output, __half* lhs, __half* rhs, int size) {
  int idx = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  if (idx < size) {
    float val = __half2float(lhs[idx]);
    output[idx] = __float2half(val / (1.0f + expf(-val)) * __half2float(rhs[idx]));
  }
}
