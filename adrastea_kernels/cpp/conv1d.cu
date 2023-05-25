#include "compat.h"

// literally the slowest possible conv1d implementation. reference.
extern "C" __global__ void conv1d(__half* output,
                                  __half* input,
                                  __half* weights,
                                  __half* bias,
                                  int channels_in,
                                  int channels_out,
                                  int kernel_size,
                                  int width,
                                  int stride,
                                  int padding) {
  int c = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int r = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  int out_width = GRID_DIM_X * BLOCK_DIM_X;
  if (r >= channels_out || c >= out_width) {
    return;
  }
  float sum = 0.0;
  int start_x = c * stride - padding;
  int end_x = start_x + kernel_size;
  for (int c_in = 0; c_in < channels_in; ++c_in) {
    for (int x = max(0, start_x); x < min(width, end_x); ++x) {
      sum +=
          __half2float(input[c_in * width + x]) *
          __half2float(weights[r * channels_in * kernel_size + c_in * kernel_size + x - start_x]);
    }
  }
  output[r * out_width + c] = sum + __half2float(bias[r]);
}