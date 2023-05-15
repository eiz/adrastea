#include "compat.h"

extern "C" __global__ void rotary(__half* output,
                                  __half* input,
                                  int h,
                                  int w,
                                  int n_heads,
                                  int pos_offset = 0,
                                  float theta = 10000.0) {
  int r = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  int c = 2 * (BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X);
  int head_dim = w / n_heads;
  int head_c = c % head_dim;

  if (r < h && c < w) {
    float angle = (pos_offset + r) / powf(theta, float(head_c) / head_dim);
    float real = __half2float(input[r * w + c]);
    float imag = __half2float(input[r * w + c + 1]);
    float a_cos = cosf(angle);
    float a_sin = sinf(angle);
    output[r * w + c] = __float2half(real * a_cos - imag * a_sin);
    output[r * w + c + 1] = __float2half(real * a_sin + imag * a_cos);
  }
}
