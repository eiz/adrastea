#include "compat.h"

// input operators:
//  - half to float
//  - scale by constant
// output operators:
//  - gelu activation
//  - bias add
//  - beta
//
// order is uhhh beta(gelu(bias))
template <typename T, typename O = T>
struct Identity {
 public:
  __device__ __forceinline__ T operator()(T const value) const { return value; }
  __device__ __forceinline__ T operator()(T const value,
                                          O const* const output,
                                          int x,
                                          int y,
                                          int stride_ox,
                                          int stride_oy) const {
    return value;
  }
};

template <typename T, typename Operator = Identity<T>>
struct Scale {
 public:
  __device__ __forceinline__ Scale(T scale, Operator op) : scale(scale), op(op) {}
  __device__ __forceinline__ T operator()(T const value) const { return op(value) * scale; }

  T scale;
  Operator op;
};

template <typename Operator = Identity<half>>
struct HalfToFloat {
  __device__ __forceinline__ float operator()(__half const& x, Operator op) const {
    return __half2float(op(x));
  }
};

template <typename T, typename O, typename Operator = Identity<T>>
struct Bias {
  __device__ __forceinline__ Bias(T const* bias, Operator op) : bias(bias), op(op) {}
  __device__ __forceinline__ T operator()(T const value,
                                          O const* const output,
                                          int x,
                                          int y,
                                          int stride_ox,
                                          int stride_oy) const {
    return op(value, x, y, stride_ox, stride_oy) + bias[x];
  }

  T const* bias;
  Operator op;
};

template <typename T, typename O, typename Operator = Identity<T>>
struct Gelu {
  __device__ __forceinline__ Gelu(Operator op) : op(op) {}
  __device__ __forceinline__ T operator()(T const value,
                                          O const* const output,
                                          int x,
                                          int y,
                                          int stride_ox,
                                          int stride_oy) const {
    auto inner = op(value, x, y, stride_ox, stride_oy);
    return 0.5f * inner *
           (1.0f + tanhf(0.7978845608028654f * (inner + 0.044715f * inner * inner * inner)));
  }

  Operator op;
};

template <typename T, typename O, typename Operator = Identity<T>>
struct Beta {
  __device__ __forceinline__ Beta(T beta, Operator op) : beta(beta), op(op) {}
  __device__ __forceinline__ T operator()(T const value,
                                          O const* const output,
                                          int x,
                                          int y,
                                          int stride_ox,
                                          int stride_oy) const {
    return op(value, x, y, stride_ox, stride_oy) + beta * output[x * stride_oy + y * stride_ox];
  }

  T beta;
  Operator op;
};

template <typename T, typename InputOperator = Identity<T>, typename OutputOperator = Identity<T>>
void __device__ __forceinline__ matmul(T* const output,
                                       T const* const lhs,
                                       T const* const rhs,
                                       int const batches,
                                       int const m,
                                       int const k,
                                       int const n,
                                       int const stride_oz,
                                       int const stride_ox,
                                       int const stride_oy,
                                       int const stride_lz,
                                       int const stride_lx,
                                       int const stride_ly,
                                       int const stride_rz,
                                       int const stride_rx,
                                       int const stride_ry,
                                       InputOperator input_operator = {},
                                       OutputOperator output_operator = {}) {
  int c = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int r = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  int b = BLOCK_IDX_Z * BLOCK_DIM_Z + THREAD_IDX_Z;
  output += b * stride_oz;
  lhs += b * stride_lz;
  rhs += b * stride_rz;
  if (r < m && c < n) {
    T sum = 0;
    for (int i = 0; i < k; i++) {
      sum += input_operator(lhs[r * stride_ly + i * stride_lx], rhs[i * stride_ry + c * stride_rx]);
    }
    output[r * stride_oy + c * stride_ox] =
        output_operator(sum, output, c, r, stride_ox, stride_oy);
  }
}

// output = lhs * rhs^T + bias + output * beta; lhs = (m, k); rhs = (n, k); output = (m, n)
// this only exists as a trivial reference point for optimized kernels.
extern "C" __global__ void linear(__half* const output,
                                  __half const* const lhs,
                                  __half const* const rhs,
                                  __half const* const bias,
                                  int const m,
                                  int const k,
                                  int const n,
                                  int const stride_ox,
                                  int const stride_oy,
                                  int const stride_lx,
                                  int const stride_ly,
                                  int const stride_rx,
                                  int const stride_ry,
                                  float const beta = 0.0f) {
  int c = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int r = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  if (r < m && c < n) {
    float sum = 0;
    for (int i = 0; i < k; i++) {
      sum += __half2float(lhs[r * stride_ly + i * stride_lx]) *
             __half2float(rhs[c * stride_ry + i * stride_rx]);
    }
    output[r * stride_oy + c * stride_ox] =
        sum + __half2float(bias ? bias[c] : __half(0.0)) +
        beta * __half2float(output[r * stride_oy + c * stride_ox]);
  }
}
