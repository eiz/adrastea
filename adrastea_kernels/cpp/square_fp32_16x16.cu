#include "compat.h"

extern "C" __global__ void square_fp32_16x16(float* output, float* input, int width, int height) {
  int x = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int y = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  int idx = y * width + x;
  if (x < width && y < height) {
    output[idx] = input[idx] * input[idx];
  }
}