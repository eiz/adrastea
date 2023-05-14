#include <cstdint>

#include <cuda_fp16.h>

// the simplest quantized right side kernel. reference.
extern "C" __global__ void matmul_nt_fp16u8(half* __restrict__ output,
                                            half const* __restrict__ lhs,
                                            uint8_t const* __restrict__ rhs,
                                            half const* __restrict__ scales,
                                            int m,
                                            int p,
                                            int n,
                                            int block_size,
                                            float beta = 0.0f) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int block_count = p / block_size;
  if (r < m && c < n) {
    float sum = 0;
    for (int i = 0; i < p; i++) {
      sum += __half2float(lhs[r * p + i]) * (float(rhs[c * p + i]) - 127.5) *
             __half2float(scales[c * block_count + i / block_size]);
    }
    output[r * n + c] = sum + beta * __half2float(output[r * n + c]);
  }
}
