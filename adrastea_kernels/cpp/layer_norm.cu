#include "compat.h"

// row-wise layer normalization
// 1 block per row, x = row, 8 warps per block
// TODO strides kill the performance, refuxor later
extern "C" __global__ void layer_norm(__half* output,
                                      __half* input,
                                      __half* weights,
                                      __half* bias,
                                      int length_x,
                                      int length_y,
                                      int stride_ox,
                                      int stride_oy,
                                      int stride_ix,
                                      int stride_iy,
                                      float eps) {
  int row = BLOCK_IDX_X;
  int tid = THREAD_IDX_X;
  int in_row_idx = row * stride_iy;
  int out_row_idx = row * stride_oy;
  int warp_id = tid / 32;
  bool warp_leader = (tid % 32) == 0;
  __shared__ float s_mean;
  __shared__ float s_stddev;
  __shared__ float s_warp_reduced[8];
  float sum_val = 0.0f;
  // sum: thread reduction
  for (int i = tid; i < length_x; i += BLOCK_DIM_X) {
    float val = __half2float(input[in_row_idx + i * stride_ix]);
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
      s_mean = sum_val / length_x;
    }
  }
  __syncthreads();
  sum_val = 0.0f;
  // mean diff: thread reduction
  for (int i = tid; i < length_x; i += BLOCK_DIM_X) {
    float val = __half2float(input[in_row_idx + i * stride_ix]);
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
      s_stddev = sqrt(sum_val / length_x + eps);
    }
  }
  __syncthreads();
  for (int i = tid; i < length_x; i += BLOCK_DIM_X) {
    output[out_row_idx + i * stride_ox] =
        (__half2float(input[in_row_idx + i * stride_ix]) - s_mean) / s_stddev *
            __half2float(weights[i]) +
        __half2float(bias[i]);
  }
}
