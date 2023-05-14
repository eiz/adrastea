#include <cuda_fp16.h>

// output = lhs * rhs^T; lhs = (m, p); rhs = (n, p); output = (m, n)
// this only exists as a trivial reference point for optimized kernels.
extern "C" __global__ void
matmul_nt(__half* output, __half* lhs, __half* rhs, int m, int p, int n, float beta = 0.0f) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < m && c < n) {
    float sum = 0;
    for (int i = 0; i < p; i++) {
      sum += __half2float(lhs[r * p + i]) * __half2float(rhs[c * p + i]);
    }
    output[r * n + c] = sum + beta * __half2float(output[r * n + c]);
  }
}
