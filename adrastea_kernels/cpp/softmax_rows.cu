#include <cuda_fp16.h>
#include <math_constants.h>

// row-wise softmax with optional temperature
// input = (n_heads, seq_len, seq_len)
// output = (n_heads, seq_len, seq_len)
extern "C" __global__ void softmax_rows(__half* output,
                                        __half* input,
                                        int h,
                                        int w,
                                        float temp = 1.0) {
  int row = blockIdx.x;
  int head = blockIdx.y;
  int tid = threadIdx.x;
  int row_idx = head * h * w + row * w;
  bool warp_leader = tid % 32 == 0;
  int warp_id = tid / 32;
  __shared__ float s_max_val, s_sum_exp;
  __shared__ float s_warp_reduced[8];
  // max: thread reduction
  float max_val = -CUDART_INF_F;
  for (int i = tid; i < w; i += blockDim.x) {
    max_val = fmaxf(max_val, __half2float(input[row_idx + i]) / temp);
  }
  __syncthreads();
  // max: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_xor_sync(~0, max_val, offset);
    max_val = fmaxf(max_val, other_val);
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = max_val;
  }
  // max: block reduction
  __syncthreads();
  if (warp_id == 0) {
    max_val = (tid < 8) ? s_warp_reduced[tid] : -CUDART_INF_F;
    for (int offset = 4; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, max_val, offset);
      max_val = fmaxf(max_val, other_val);
    }
    if (warp_leader) {
      s_max_val = max_val;
    }
  }
  __syncthreads();
  float sum_val = 0.0f;
  // expsum: thread reduction
  for (int i = tid; i < w; i += blockDim.x) {
    float val = __half2float(input[row_idx + i]) / temp;
    sum_val += expf(val - s_max_val);
  }
  __syncthreads();
  // expsum: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_val += __shfl_xor_sync(~0, sum_val, offset);
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = sum_val;
  }
  // expsum: block reduction
  __syncthreads();
  if (warp_id == 0) {
    sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
    for (int offset = 4; offset > 0; offset /= 2) {
      sum_val += __shfl_xor_sync(~0, sum_val, offset);
    }
    if (warp_leader) {
      s_sum_exp = sum_val;
    }
  }
  __syncthreads();
  for (int i = tid; i < w; i += blockDim.x) {
    float val = __half2float(input[row_idx + i]) / temp;
    output[row_idx + i] = __float2half(expf(val - s_max_val) / s_sum_exp);
  }
}
