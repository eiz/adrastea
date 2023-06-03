#include "compat.h"

enum ActivationType { IDENTITY, GELU };

struct IdentityActivation {
  __device__ __forceinline__ float operator()(float x) { return x; }
};

struct GeluActivation {
  __device__ __forceinline__ float operator()(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  }
};

// literally the slowest possible conv1d implementation. reference.
// TODO fix the strides/dimensions convention on this thing
template <typename Activation>
__device__ void conv1d(__half* output,
                       __half* input,
                       __half* weights,
                       __half* bias,
                       int channels_in,
                       int channels_out,
                       int kernel_size,
                       int width,
                       int out_width,
                       int stride,
                       int padding) {
  Activation activation;
  int c = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int r = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  if (r >= channels_out || c >= out_width) {
    return;
  }
  float sum = __half2float(bias[r]);
  int start_x = c * stride - padding;
  int end_x = start_x + kernel_size;
  for (int c_in = 0; c_in < channels_in; ++c_in) {
    for (int x = max(0, start_x); x < min(width, end_x); ++x) {
      sum +=
          __half2float(input[c_in * width + x] *
                       weights[r * channels_in * kernel_size + c_in * kernel_size + x - start_x]);
    }
  }
  output[r * out_width + c] = activation(sum);
}

extern "C" __global__ void conv1d(__half* output,
                                  __half* input,
                                  __half* weights,
                                  __half* bias,
                                  int channels_in,
                                  int channels_out,
                                  int kernel_size,
                                  int width,
                                  int out_width,
                                  int stride,
                                  int padding,
                                  ActivationType activation) {
  if (activation == IDENTITY) {
    conv1d<IdentityActivation>(output, input, weights, bias, channels_in, channels_out, kernel_size,
                               width, out_width, stride, padding);
  } else if (activation == GELU) {
    conv1d<GeluActivation>(output, input, weights, bias, channels_in, channels_out, kernel_size,
                           width, out_width, stride, padding);
  }
}
