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
template <typename Activation>
__device__ void conv1d(TensorViewF16 output,
                       TensorViewF16 input,
                       TensorViewF16 weights,
                       TensorViewF16 bias,
                       int stride,
                       int padding) {
  Activation activation;
  int c = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int r = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  if (r >= output.height() || c >= output.width()) {
    return;
  }
  float sum = __half2float(bias(r));
  int start_x = c * stride - padding;
  int end_x = start_x + weights.width();
  for (int c_in = 0; c_in < input.height(); ++c_in) {
    for (int x = max(0, start_x); x < min(input.width(), end_x); ++x) {
      sum += __half2float(input(c_in, x) * weights(r, c_in, x - start_x));
    }
  }
  output(r, c) = activation(sum);
}

extern "C" __global__ void conv1d(TensorViewF16 output,
                                  TensorViewF16 input,
                                  TensorViewF16 weights,
                                  TensorViewF16 bias,
                                  int stride,
                                  int padding,
                                  ActivationType activation) {
  if (activation == IDENTITY) {
    conv1d<IdentityActivation>(output, input, weights, bias, stride, padding);
  } else if (activation == GELU) {
    conv1d<GeluActivation>(output, input, weights, bias, stride, padding);
  }
}

// broken kernel
template <typename T, typename Activation>
__device__ void conv2d(TensorView<T> output,
                       TensorView<T> input,
                       TensorView<T> weights,
                       TensorView<T> bias,
                       int stride_x,
                       int stride_y,
                       int padding_x,
                       int padding_y) {
  Activation activation;
  int ox = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int oy = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  int oc = BLOCK_IDX_Z * BLOCK_DIM_Z + THREAD_IDX_Z;
  if (oc >= output.channels() || oy >= output.height() || ox >= output.width()) {
    return;
  }
  float sum = float(bias(oc));
  int start_x = ox * stride_x - padding_x;
  int end_x = start_x + weights.width();
  int start_y = oy * stride_y - padding_y;
  int end_y = start_y + weights.height();
  for (int c_in = 0; c_in < input.channels(); ++c_in) {
    for (int y = max(0, start_y); y < min(input.height(), end_y); ++y) {
      for (int x = max(0, start_x); x < min(input.width(), end_x); ++x) {
        sum += float(input(c_in, y, x) * weights(oy, c_in, y - start_y, x - start_x));
      }
    }
  }
  output(oc, oy, ox) = activation(sum);
}

extern "C" __global__ void conv2d_f16(TensorViewF16 output,
                                      TensorViewF16 input,
                                      TensorViewF16 weights,
                                      TensorViewF16 bias,
                                      int stride_x,
                                      int stride_y,
                                      int padding_x,
                                      int padding_y,
                                      ActivationType activation) {
  if (activation == IDENTITY) {
    conv2d<half, IdentityActivation>(output, input, weights, bias, stride_x, stride_y, padding_x,
                                     padding_y);
  } else if (activation == GELU) {
    conv2d<half, GeluActivation>(output, input, weights, bias, stride_x, stride_y, padding_x,
                                 padding_y);
  }
}

extern "C" __global__ void conv2d_f32(TensorViewF32 output,
                                      TensorViewF32 input,
                                      TensorViewF32 weights,
                                      TensorViewF32 bias,
                                      int stride_x,
                                      int stride_y,
                                      int padding_x,
                                      int padding_y,
                                      ActivationType activation) {
  if (activation == IDENTITY) {
    conv2d<float, IdentityActivation>(output, input, weights, bias, stride_x, stride_y, padding_x,
                                      padding_y);
  } else if (activation == GELU) {
    conv2d<float, GeluActivation>(output, input, weights, bias, stride_x, stride_y, padding_x,
                                  padding_y);
  }
}
