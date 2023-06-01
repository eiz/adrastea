#include "compat.h"

#include <cassert>

// input operators:
//  - half to float
//  - scale by constant
// output operators:
//  - gelu activation
//  - bias add
//  - beta
//
// order is uhhh beta(gelu(bias))

enum class MatmulLoadOp { IDENTITY = 0, SCALE = 1 };
enum class MatmulStoreOp {
  IDENTITY = 0,
  GELU_BIAS = 1,
  BETA_GELU_BIAS = 2,
};

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
  __device__ __forceinline__ HalfToFloat(Operator op) : op(op) {}
  __device__ __forceinline__ float operator()(__half const& x) const { return __half2float(op(x)); }
  __device__ __forceinline__ float operator()(__half const& value,
                                              __half const* const output,
                                              int x,
                                              int y,
                                              int stride_ox,
                                              int stride_oy) const {
    return __half2float(op(value, output, x, y, stride_ox, stride_oy));
  }

  Operator op;
};

template <typename T, typename O, typename Operator = Identity<T>>
struct Bias {
  __device__ __forceinline__ Bias(O const* bias, Operator op) : bias(bias), op(op) {}
  __device__ __forceinline__ T operator()(T const value,
                                          O const* const output,
                                          int x,
                                          int y,
                                          int stride_ox,
                                          int stride_oy) const {
    // TODO: hardcoded half2float
    return op(value, output, x, y, stride_ox, stride_oy) + __half2float(bias[x]);
  }

  O const* bias;
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
    auto inner = op(value, output, x, y, stride_ox, stride_oy);
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
    return op(value, output, x, y, stride_ox, stride_oy) +
           beta * output[x * stride_oy + y * stride_ox];
  }

  T beta;
  Operator op;
};

template <typename T, typename InputOperator = Identity<T>, typename OutputOperator = Identity<T>>
void __device__ __forceinline__ matmul(T* output,
                                       T const* lhs,
                                       T const* rhs,
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
      sum += input_operator(lhs[r * stride_ly + i * stride_lx]) *
             input_operator(rhs[i * stride_ry + c * stride_rx]);
    }
    output[r * stride_oy + c * stride_ox] =
        output_operator(sum, output, c, r, stride_ox, stride_oy);
  }
}

template <typename T>
void __device__ __forceinline__ matmul_generic(T* const output,
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
                                               T const* const bias,
                                               float const beta = 0.0f,
                                               float const scale = 1.0f,
                                               MatmulStoreOp store_op = MatmulStoreOp::IDENTITY,
                                               MatmulLoadOp load_op = MatmulLoadOp::IDENTITY) {
  switch (store_op) {
    case MatmulStoreOp::GELU_BIAS: {
      // TODO: half2float here breaks for other T
      Gelu<float, T, Bias<float, T, HalfToFloat<>>> store_op{
          Bias<float, T, HalfToFloat<>>(bias, HalfToFloat<>(Identity<T>()))};
      switch (load_op) {
        case MatmulLoadOp::IDENTITY:
          matmul(output, lhs, rhs, batches, m, k, n, stride_oz, stride_ox, stride_oy, stride_lz,
                 stride_lx, stride_ly, stride_rz, stride_rx, stride_ry, {}, store_op);
          break;
        case MatmulLoadOp::SCALE:
          matmul(output, lhs, rhs, batches, m, k, n, stride_oz, stride_ox, stride_oy, stride_lz,
                 stride_lx, stride_ly, stride_rz, stride_rx, stride_ry,
                 Scale<float, HalfToFloat<>>(scale, HalfToFloat<>(Identity<T>())), store_op);
          break;
        default:
          assert(false && "nyi");
      }
      break;
    }
    default:
      assert(false && "nyi");
  }
}

extern "C" __global__ void matmul_f16(__half* const output,
                                      __half const* const lhs,
                                      __half const* const rhs,
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
                                      __half const* const bias,
                                      float const beta = 0.0f,
                                      float const scale = 1.0f,
                                      MatmulStoreOp store_op = MatmulStoreOp::IDENTITY,
                                      MatmulLoadOp load_op = MatmulLoadOp::IDENTITY) {
  matmul_generic(output, lhs, rhs, batches, m, k, n, stride_oz, stride_ox, stride_oy, stride_lz,
                 stride_lx, stride_ly, stride_rz, stride_rx, stride_ry, bias, beta, scale, store_op,
                 load_op);
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
