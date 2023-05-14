#include <cuda_fp16.h>

extern "C" __global__ void silu(__half* output, __half* lhs, __half* rhs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = __half2float(lhs[idx]);
    output[idx] = __float2half(val / (1.0f + expf(-val)) * __half2float(rhs[idx]));
  }
}
