#include "compat.h"

// row-wise layer normalization
// 1 block per row, x = row, 8 warps per block
extern "C" __global__ void
layer_norm(__half* output, __half* input, __half* weights, __half* bias, int h, int w, float eps) {
  int row = BLOCK_IDX_X;
  int tid = THREAD_IDX_X;
  int row_idx = row * w;
  int warp_id = tid / 32;
  bool warp_leader = (tid % 32) == 0;
  __shared__ float s_mean;
  __shared__ float s_stddev;
  __shared__ float s_warp_reduced[8];
  float sum_val = 0.0f;
  // sum: thread reduction
  for (int i = tid; i < w; i += BLOCK_DIM_X) {
    float val = __half2float(input[row_idx + i]);
    sum_val += val;
  }
  __syncthreads();
  // sum: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_xor_sync(~0, sum_val, offset);
    sum_val += other_val;
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = sum_val;
  }
  // sum: block reduction
  __syncthreads();
  if (warp_id == 0) {
    sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
    for (int offset = 4; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, sum_val, offset);
      sum_val += other_val;
    }
    if (warp_leader) {
      s_mean = sum_val / w;
    }
  }
  __syncthreads();
  sum_val = 0.0f;
  // mean diff: thread reduction
  for (int i = tid; i < w; i += BLOCK_DIM_X) {
    float val = __half2float(input[row_idx + i]);
    sum_val += (val - s_mean) * (val - s_mean);
  }
  __syncthreads();
  // mean diff: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_xor_sync(~0, sum_val, offset);
    sum_val += other_val;
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = sum_val;
  }
  // mean diff: block reduction
  __syncthreads();
  if (warp_id == 0) {
    sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
    for (int offset = 4; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, sum_val, offset);
      sum_val += other_val;
    }
    if (warp_leader) {
      s_stddev = sqrt(sum_val + eps);
    }
  }
  __syncthreads();
  for (int i = tid; i < w; i += BLOCK_DIM_X) {
    output[row_idx + i] = __half2float(input[row_idx + i]) / s_stddev * __half2float(weights[i]) +
                          __half2float(bias[i]);
  }
}
