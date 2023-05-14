#include <cuda_fp16.h>

// row-wise rms normalization
// 1 block per row, x = row, 8 warps per block
extern "C" __global__ void rms_norm(__half* output,
                                    __half* input,
                                    __half* weights,
                                    int h,
                                    int w,
                                    float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int row_idx = row * w;
  int warp_id = tid / 32;
  bool warp_leader = (tid % 32) == 0;
  __shared__ float s_rms_inv;
  __shared__ float s_warp_reduced[8];
  float sum_val = 0.0f;
  // sum_sq: thread reduction
  for (int i = tid; i < w; i += blockDim.x) {
    float val = __half2float(input[row_idx + i]);
    sum_val += val * val;
  }
  __syncthreads();
  // sum_sq: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_xor_sync(~0, sum_val, offset);
    sum_val += other_val;
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = sum_val;
  }
  // sum_sq: block reduction
  __syncthreads();
  if (warp_id == 0) {
    sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
    for (int offset = 4; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, sum_val, offset);
      sum_val += other_val;
    }
    if (warp_leader) {
      s_rms_inv = rsqrt((sum_val / w) + eps);
    }
  }
  __syncthreads();
  float rms_inv = s_rms_inv;
  for (int i = tid; i < w; i += blockDim.x) {
    float val = __half2float(input[row_idx + i]);
    output[row_idx + i] = weights[i] * __float2half(val * rms_inv);
  }
}
