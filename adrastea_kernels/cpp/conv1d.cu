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

// tryin something out here
// low 3 bits of ptr = n_dims
// strides/shape are in reverse order (the dimension normally known as width is shape[0])
template <typename T>
struct TensorView {
  uint64_t ptr;
  uint32_t shape[7];
  uint32_t strides[7];

  __device__ __forceinline__ T* as_ptr() { return reinterpret_cast<T*>(ptr & ~7); }
  __device__ __forceinline__ size_t n_dims() { return ptr & 7; }

  // traditional NCHW dimension names
  __device__ __forceinline__ uint32_t width() { return shape[0]; }
  __device__ __forceinline__ uint32_t height() { return shape[1]; }
  __device__ __forceinline__ uint32_t channels() { return shape[2]; }
  __device__ __forceinline__ uint32_t batches() { return shape[3]; }

  __device__ __forceinline__ T& operator()(uint32_t x) { return as_ptr()[x * strides[0]]; }
  __device__ __forceinline__ T& operator()(uint32_t y, uint32_t x) {
    return as_ptr()[y * strides[1] + x * strides[0]];
  }
  __device__ __forceinline__ T& operator()(uint32_t c, uint32_t y, uint32_t x) {
    return as_ptr()[c * strides[2] + y * strides[1] + x * strides[0]];
  }
  __device__ __forceinline__ T& operator()(uint32_t n, uint32_t c, uint32_t y, uint32_t x) {
    return as_ptr()[n * strides[3] + c * strides[2] + y * strides[1] + x * strides[0]];
  }
};

using TensorViewF16 = TensorView<__half>;

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
