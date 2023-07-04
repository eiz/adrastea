#include "compat.h"

#include <cassert>

enum UnaryOp {
  IDENTITY = 1,
};

struct Identity {
  template <typename T>
  __device__ __forceinline__ T operator()(T const& input) const {
    return input;
  }
};

enum BinaryOp {
  ADD = 1,
  SILU_MUL = 2,
};

struct Add {
  template <typename T>
  __device__ __forceinline__ T operator()(T const& left, T const& right) const {
    return left + right;
  }
};

struct SiluMul {
  template <typename T>
  __device__ __forceinline__ T operator()(T const& left, T const& right) const {
    return T(float(left) / (1.0f + expf(-left)) * float(right));
  }
};

template <typename T, typename Operator>
void __device__ __forceinline__ elementwise_unary_1d(TensorView<T> output,
                                                     TensorView<T> input,
                                                     int const i) {
  using Frame = typename TensorView<T>::StackFrame;
  output.iter_dims(0, [&output, &input, i](Frame* tos, Frame* bos) {
    Operator op;
    auto output_slice = output.slice(tos, bos);
    auto input_slice = input.slice(tos, bos);
    output_slice(i) = op(input_slice(i));
  });
}

template <typename T, typename Operator>
void __device__ __forceinline__
elementwise_unary_2d(TensorView<T> output, TensorView<T> input, int const i, int const j) {
  using Frame = typename TensorView<T>::StackFrame;
  output.iter_dims(1, [&output, &input, i, j](Frame* tos, Frame* bos) {
    Operator op;
    auto output_slice = output.slice(tos, bos);
    auto input_slice = input.slice(tos, bos);
    output_slice(j, i) = op(input_slice(j, i));
  });
}

template <typename T, typename Operator>
void __device__ __forceinline__
elementwise_binary_1d(TensorView<T> output, TensorView<T> left, TensorView<T> right, int const i) {
  using Frame = typename TensorView<T>::StackFrame;
  output.iter_dims(0, [&output, &left, &right, i](Frame* tos, Frame* bos) {
    Operator op;
    auto output_slice = output.slice(tos, bos);
    auto left_slice = left.slice(tos, bos);
    auto right_slice = right.slice(tos, bos);
    output_slice(i) = op(left_slice(i), right_slice(i));
  });
}

template <typename T, typename Operator>
void __device__ __forceinline__ elementwise_binary_2d(TensorView<T> output,
                                                      TensorView<T> left,
                                                      TensorView<T> right,
                                                      int const i,
                                                      int const j) {
  using Frame = typename TensorView<T>::StackFrame;
  output.iter_dims(1, [&output, &left, &right, i, j](Frame* tos, Frame* bos) {
    Operator op;
    auto output_slice = output.slice(tos, bos);
    auto left_slice = left.slice(tos, bos);
    auto right_slice = right.slice(tos, bos);
    output_slice(j, i) = op(left_slice(j, i), right_slice(j, i));
  });
}

template <typename T>
void __device__ __forceinline__ elementwise_unary_1d_generic(TensorView<T> output,
                                                             TensorView<T> input,
                                                             UnaryOp op) {
  int n = BLOCK_DIM_X * BLOCK_IDX_X + THREAD_IDX_X;
  if (n >= output.width())
    return;

  switch (op) {
    case IDENTITY:
      elementwise_unary_1d<T, Identity>(output, input, n);
      break;
    default:
      assert(false);
  }
}

template <typename T>
void __device__ __forceinline__ elementwise_unary_2d_generic(TensorView<T> output,
                                                             TensorView<T> input,
                                                             UnaryOp op) {
  int x = BLOCK_DIM_X * BLOCK_IDX_X + THREAD_IDX_X;
  int y = BLOCK_DIM_Y * BLOCK_IDX_Y + THREAD_IDX_Y;
  if (x >= output.width() || y >= output.height())
    return;

  switch (op) {
    case IDENTITY:
      elementwise_unary_2d<T, Identity>(output, input, x, y);
      break;
    default:
      assert(false);
  }
}

template <typename T>
void __device__ __forceinline__ elementwise_binary_1d_generic(TensorView<T> output,
                                                              TensorView<T> left,
                                                              TensorView<T> right,
                                                              BinaryOp op) {
  int n = BLOCK_DIM_X * BLOCK_IDX_X + THREAD_IDX_X;
  if (n >= output.width())
    return;

  switch (op) {
    case ADD:
      elementwise_binary_1d<T, Add>(output, left, right, n);
      break;
    case SILU_MUL:
      elementwise_binary_1d<T, SiluMul>(output, left, right, n);
      break;
    default:
      assert(false);
  }
}

template <typename T>
void __device__ __forceinline__ elementwise_binary_2d_generic(TensorView<T> output,
                                                              TensorView<T> left,
                                                              TensorView<T> right,
                                                              BinaryOp op) {
  int x = BLOCK_DIM_X * BLOCK_IDX_X + THREAD_IDX_X;
  int y = BLOCK_DIM_Y * BLOCK_IDX_Y + THREAD_IDX_Y;
  if (x >= output.width() || y >= output.height())
    return;

  switch (op) {
    case ADD:
      elementwise_binary_2d<T, Add>(output, left, right, x, y);
      break;
    case SILU_MUL:
      elementwise_binary_2d<T, SiluMul>(output, left, right, x, y);
      break;
    default:
      assert(false);
  }
}

extern "C" {
void __global__ elementwise_unary_1d_f16(TensorViewF16 output, TensorViewF16 input, UnaryOp op) {
  elementwise_unary_1d_generic<half>(output, input, op);
}

void __global__ elementwise_unary_2d_f16(TensorViewF16 output, TensorViewF16 input, UnaryOp op) {
  elementwise_unary_2d_generic<half>(output, input, op);
}

void __global__ elementwise_unary_2d_f32(TensorViewF32 output, TensorViewF32 input, UnaryOp op) {
  elementwise_unary_2d_generic<float>(output, input, op);
}

void __global__ elementwise_binary_1d_f16(TensorViewF16 output,
                                          TensorViewF16 left,
                                          TensorViewF16 right,
                                          BinaryOp op) {
  elementwise_binary_1d_generic<half>(output, left, right, op);
}

void __global__ elementwise_binary_1d_f32(TensorViewF32 output,
                                          TensorViewF32 left,
                                          TensorViewF32 right,
                                          BinaryOp op) {
  elementwise_binary_1d_generic<float>(output, left, right, op);
}

void __global__ elementwise_binary_2d_f16(TensorViewF16 output,
                                          TensorViewF16 left,
                                          TensorViewF16 right,
                                          BinaryOp op) {
  elementwise_binary_2d_generic<half>(output, left, right, op);
}

void __global__ elementwise_binary_2d_f32(TensorViewF32 output,
                                          TensorViewF32 left,
                                          TensorViewF32 right,
                                          BinaryOp op) {
  elementwise_binary_2d_generic<float>(output, left, right, op);
}
}  // extern "C"
