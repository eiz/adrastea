#include "compat.h"

// TODO: this kernel should be fused out of existence.
extern "C" __global__ void rotary(__half* output,
                                  __half* input,
                                  int length_y,
                                  int length_x,
                                  int stride_ox,
                                  int stride_oy,
                                  int stride_ix,
                                  int stride_iy,
                                  int n_heads,
                                  int pos_offset,
                                  float theta) {
  int r = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  int c = 2 * (BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X);
  int head_dim = length_x / n_heads;
  int head_c = c % head_dim;

  if (r < length_y && c < length_x) {
    float angle = (pos_offset + r) / powf(theta, float(head_c) / head_dim);
    float real = __half2float(input[r * stride_iy + c * stride_ix]);
    float imag = __half2float(input[r * stride_iy + (c + 1) * stride_ix]);
    float a_cos = cosf(angle);
    float a_sin = sinf(angle);
    output[r * stride_oy + c * stride_ox] = __float2half(real * a_cos - imag * a_sin);
    output[r * stride_oy + (c + 1) * stride_ox] = __float2half(real * a_sin + imag * a_cos);
  }
}
