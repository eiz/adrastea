#include "compat.h"

#include <cassert>

// TODO this is probably stupid and doesn't work, check the asm
// TODO ensure contiguous memory access between adjacent threads in strided cases
#define STRIDE_1(stride, ...) \
  if (stride == 1) {          \
    __VA_ARGS__;              \
  } else {                    \
    __VA_ARGS__;              \
  }

#define STRIDE_1_BINARY_1D(stride_1, stride_2, stride_3, ...) \
  STRIDE_1(stride_1, STRIDE_1(stride_2, STRIDE_1(stride_3, __VA_ARGS__)))

#define STRIDE_1_BINARY_2D(stride_1x, stride_1y, stride_2x, stride_2y, stride_3x, stride_3y, ...) \
  STRIDE_1_BINARY_1D(stride_1x, stride_2x, stride_3x,                                             \
                     STRIDE_1_BINARY_1D(stride_1y, stride_2y, stride_3y, __VA_ARGS__))

enum BinaryOp {
  ADD = 1,
};

struct Add {
  template <typename T>
  __device__ __forceinline__ T operator()(T const& left, T const& right) const {
    return left + right;
  }
};

template <typename T, typename Operator>
void __device__ __forceinline__ elementwise_binary_1d(T* const output,
                                                      T const* const left,
                                                      T const* const right,
                                                      int const stride_o,
                                                      int const stride_l,
                                                      int const stride_r,
                                                      int const i) {
  Operator op;
  output[i * stride_o] = op(left[i * stride_l], right[i * stride_r]);
}

template <typename T, typename Operator>
void __device__ __forceinline__ elementwise_binary_2d(T* const output,
                                                      T const* const left,
                                                      T const* const right,
                                                      int const stride_ox,
                                                      int const stride_oy,
                                                      int const stride_lx,
                                                      int const stride_ly,
                                                      int const stride_rx,
                                                      int const stride_ry,
                                                      int const i,
                                                      int const j) {
  Operator op;
  output[i * stride_ox + j * stride_oy] =
      op(left[i * stride_lx + j * stride_ly], right[i * stride_rx + j * stride_ry]);
}

template <typename T>
void __device__ __forceinline__ elementwise_binary_1d_generic(T* const output,
                                                              T const* const left,
                                                              T const* const right,
                                                              int length,
                                                              int const stride_o,
                                                              int const stride_l,
                                                              int const stride_r,
                                                              BinaryOp op) {
  int n = BLOCK_DIM_X * BLOCK_IDX_X + THREAD_IDX_X;
  if (n >= length)
    return;

  switch (op) {
    case ADD:
      STRIDE_1_BINARY_1D(
          stride_o, stride_l, stride_r,
          elementwise_binary_1d<T, Add>(output, left, right, stride_o, stride_l, stride_r, n));
      break;
    default:
      assert(false);
  }
}

template <typename T>
void __device__ __forceinline__ elementwise_binary_2d_generic(T* const output,
                                                              T const* const left,
                                                              T const* const right,
                                                              int length_x,
                                                              int length_y,
                                                              int const stride_ox,
                                                              int const stride_oy,
                                                              int const stride_lx,
                                                              int const stride_ly,
                                                              int const stride_rx,
                                                              int const stride_ry,
                                                              BinaryOp op) {
  int n = BLOCK_DIM_X * BLOCK_IDX_X + THREAD_IDX_X;
  int m = BLOCK_DIM_Y * BLOCK_IDX_Y + THREAD_IDX_Y;
  if (n >= length_x || m >= length_y)
    return;

  switch (op) {
    case ADD:
      STRIDE_1_BINARY_2D(
          stride_ox, stride_oy, stride_lx, stride_ly, stride_rx, stride_ry,
          elementwise_binary_2d<T, Add>(output, left, right, stride_ox, stride_oy, stride_lx,
                                        stride_ly, stride_rx, stride_ry, n, m));
      break;
    default:
      assert(false);
  }
}

extern "C" {
void __global__ elementwise_binary_1d_f16(half* const output,
                                          half const* const left,
                                          half const* const right,
                                          int length,
                                          int const stride_o,
                                          int const stride_l,
                                          int const stride_r,
                                          BinaryOp op) {
  elementwise_binary_1d_generic<half>(output, left, right, length, stride_o, stride_l, stride_r,
                                      op);
}

void __global__ elementwise_binary_1d_f32(float* const output,
                                          float const* const left,
                                          float const* const right,
                                          int length,
                                          int const stride_o,
                                          int const stride_l,
                                          int const stride_r,
                                          BinaryOp op) {
  elementwise_binary_1d_generic<float>(output, left, right, length, stride_o, stride_l, stride_r,
                                       op);
}

void __global__ elementwise_binary_2d_f16(half* const output,
                                          half const* const left,
                                          half const* const right,
                                          int length_x,
                                          int length_y,
                                          int const stride_ox,
                                          int const stride_oy,
                                          int const stride_lx,
                                          int const stride_ly,
                                          int const stride_rx,
                                          int const stride_ry,
                                          BinaryOp op) {
  elementwise_binary_2d_generic<half>(output, left, right, length_x, length_y, stride_ox, stride_oy,
                                      stride_lx, stride_ly, stride_rx, stride_ry, op);
}

void __global__ elementwise_binary_2d_f32(float* const output,
                                          float const* const left,
                                          float const* const right,
                                          int length_x,
                                          int length_y,
                                          int const stride_ox,
                                          int const stride_oy,
                                          int const stride_lx,
                                          int const stride_ly,
                                          int const stride_rx,
                                          int const stride_ry,
                                          BinaryOp op) {
  elementwise_binary_2d_generic<float>(output, left, right, length_x, length_y, stride_ox,
                                       stride_oy, stride_lx, stride_ly, stride_rx, stride_ry, op);
}
}  // extern "C"
