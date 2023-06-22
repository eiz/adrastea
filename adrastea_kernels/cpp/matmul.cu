#include "compat.h"

// A matrix multiply operator which supports various handy op fusions.
// TODO: this is currently the fully unoptimized version. implement loop
// tiling, shared memory, double buffering / async load, warp mma instructions.

// input operators:
//  - half to float
//  - scale by constant
// output operators:
//  - gelu activation
//  - (row-wise) bias add
//  - beta
// masking:
//  - causal (upper right triangle of the output is replaced with -inf)
//
// order is uhhh beta(gelu(bias))

#ifdef __gfx1100__
#define NYI() asm volatile("s_trap 2")
#else
#include <cassert>
#define NYI() assert(false && "nyi")
#endif

enum class MatmulLoadOp { IDENTITY = 0, SCALE = 1 };
enum class MatmulStoreOp {
  IDENTITY = 0,
  GELU_BIAS = 1,
  BETA_GELU_BIAS = 2,
  BETA_BIAS = 3,
  SCALE = 4,
  ADD = 5,
};
enum class MatmulMaskOp { NONE = 0, CAUSAL = 1 };

__device__ __forceinline__ float to_float(float value) {
  return value;
}
__device__ __forceinline__ float to_float(__half value) {
  return __half2float(value);
}

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
  __device__ __forceinline__ float operator()(__half const& value,
                                              __half const* const output,
                                              int x,
                                              int y,
                                              int stride_ox,
                                              int stride_oy) const {
    return op(value, output, x, y, stride_ox, stride_oy) * scale;
  }
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
           beta * __half2float(output[x * stride_ox + y * stride_oy]);
  }

  T beta;
  Operator op;
};

template <typename T, typename O, typename Operator = Identity<T>>
struct Add {
  __device__ __forceinline__ Add(Operator op) : op(op) {}
  __device__ __forceinline__ T operator()(T const value,
                                          O const* const output,
                                          int x,
                                          int y,
                                          int stride_ox,
                                          int stride_oy) const {
    return op(value, output, x, y, stride_ox, stride_oy) + output[x * stride_ox + y * stride_oy];
  }

  Operator op;
};

struct NoMask {
  __device__ __forceinline__ bool operator()(int x, int y) const { return false; }
};

struct CausalMask {
  __device__ __forceinline__ bool operator()(int x, int y) const { return x > y; }
};

#define MATMUL_COMMON_PARAMS(T)                                                              \
  T *output, T const *lhs, T const *rhs, int const batches, int const m, int k, int const n, \
      int const stride_ox, int const stride_oy, int const stride_oz, int const stride_lx,    \
      int const stride_ly, int const stride_lz, int const stride_rx, int const stride_ry,    \
      int const stride_rz
#define MATMUL_COMMON_ARGS()                                                                 \
  output, lhs, rhs, batches, m, k, n, stride_ox, stride_oy, stride_oz, stride_lx, stride_ly, \
      stride_lz, stride_rx, stride_ry, stride_rz

template <typename T,
          typename InputOperator = Identity<T>,
          typename OutputOperator = Identity<T>,
          typename MaskOperator = NoMask>
void __device__ matmul(MATMUL_COMMON_PARAMS(T),
                       InputOperator input_operator = {},
                       OutputOperator output_operator = {},
                       MaskOperator mask_operator = {}) {
  int x = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int y = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  int b = BLOCK_IDX_Z * BLOCK_DIM_Z + THREAD_IDX_Z;
  output += b * stride_oz;
  lhs += b * stride_lz;
  rhs += b * stride_rz;
  if (y < m && x < n) {
    if (mask_operator(x, y)) {
      output[y * stride_oy + x * stride_ox] = -CUDART_INF_F;
      return;
    }
    float sum = 0;
    for (int i = 0; i < k; i++) {
      sum += to_float(input_operator(lhs[y * stride_ly + i * stride_lx]) *
                      input_operator(rhs[i * stride_ry + x * stride_rx]));
    }
    output[y * stride_oy + x * stride_ox] =
        output_operator(sum, output, x, y, stride_ox, stride_oy);
  }
}

#define LANE_ID THREAD_IDX_X
#define WARP_ID THREAD_IDX_Y
#define N_LANES BLOCK_DIM_X
#define N_WARPS BLOCK_DIM_Y

#ifdef __gfx1100__
#define AS_SCALAR(x) __builtin_amdgcn_readfirstlane(x)
#else
#define AS_SCALAR(x) x
#endif

// TODO not actually fast yet...
template <typename T,
          int Mtile = 16,
          int Ktile = 32,
          int Ntile = 32,
          int Warps = 4,
          int Lanes = 32,
          int AccumY = 4,
          int AccumX = 1,
          typename InputOperator = Identity<T>,
          typename OutputOperator = Identity<T>,
          typename MaskOperator = NoMask>
void __device__ matmul_fast(MATMUL_COMMON_PARAMS(T),
                            InputOperator input_operator = {},
                            OutputOperator output_operator = {},
                            MaskOperator mask_operator = {}) {
  static_assert((Mtile * Ntile) % (Warps * Lanes) == 0,
                "Mtile * Ntile must be divisible by Warps * Lanes");
  static_assert((Mtile * Ntile) / (Warps * Lanes) == AccumY * AccumX, "Invalid AccumY and AccumX");
  static_assert(AccumY == Mtile / Warps, "Invalid AccumY");
  static_assert(AccumX == Ntile / Lanes, "Invalid AccumX");
  static_assert(Mtile % Warps == 0, "Warps must divide Mtile");
  static_assert(Ntile % Lanes == 0, "Lanes must divide Ntile");
  static_assert(Ktile % Lanes == 0, "Lanes must divide Ktile");
  static_assert(Ktile % Warps == 0, "Warps must divide Ktile");
  if (N_LANES != Lanes || N_WARPS != Warps) {
    return;
  }
  extern __shared__ void* sdata[];
  const int SDATA_BASE_LHS = 0;
  const int SDATA_BASE_RHS = Mtile * (Ktile + 2);
#define SDATA(type, side, stride, d0, d1) \
  (((type*)sdata)[SDATA_BASE_##side + ((d0) * (stride)) + (d1)])
#define LHS(d0, d1) SDATA(__half, LHS, Ktile + 2, d0, d1)
#define RHS(d0, d1) SDATA(__half, RHS, Ntile + 2, d0, d1)
  int bx = BLOCK_IDX_X;
  int by = BLOCK_IDX_Y;
  int b = BLOCK_IDX_Z;
  float v_accum[AccumY][AccumX];
  int tail_h = m - by * Mtile;
  int warp_tail_h = tail_h - int(WARP_ID);
  int tail_w = n - bx * Ntile;
  int lane_tail_w = tail_w - int(LANE_ID);
  bool is_right_row_major = stride_rx == 1;
  bool is_left_row_major = stride_lx == 1;
  output += b * stride_oz;
  lhs += b * stride_lz + by * Mtile * stride_ly;
  rhs += b * stride_rz + bx * Ntile * stride_rx;
  for (int y = 0; y < AccumY; ++y) {
    for (int x = 0; x < AccumX; ++x) {
      v_accum[y][x] = 0;
    }
  }
  while (k >= Ktile) {
    T const* l_lhs = lhs + WARP_ID * stride_ly + LANE_ID * stride_lx;
    T const* l_rhs = rhs + WARP_ID * stride_ry + LANE_ID * stride_rx;
    if (tail_h < Mtile) {
      if (is_left_row_major) {
        for (int y = 0; y < Mtile; y += Warps) {
          if (y >= warp_tail_h) {
            for (int x = 0; x < Ktile; x += Lanes) {
              LHS(y + WARP_ID, x + LANE_ID) = 0;
            }
          } else {
            for (int x = 0; x < Ktile; x += Lanes) {
              LHS(y + WARP_ID, x + LANE_ID) = input_operator(l_lhs[y * stride_ly + x]);
            }
          }
        }
      } else {
        for (int y = 0; y < Mtile; y += Warps) {
          if (y >= warp_tail_h) {
            for (int x = 0; x < Ktile; x += Lanes) {
              LHS(y + WARP_ID, x + LANE_ID) = 0;
            }
          } else {
            for (int x = 0; x < Ktile; x += Lanes) {
              LHS(y + WARP_ID, x + LANE_ID) = input_operator(l_lhs[y * stride_ly + x * stride_lx]);
            }
          }
        }
      }
    } else {
      if (is_left_row_major) {
        for (int y = 0; y < Mtile; y += Warps) {
          for (int x = 0; x < Ktile; x += Lanes) {
            LHS(y + WARP_ID, x + LANE_ID) = input_operator(l_lhs[y * stride_ly + x]);
          }
        }
      } else {
        for (int y = 0; y < Mtile; y += Warps) {
          for (int x = 0; x < Ktile; x += Lanes) {
            LHS(y + WARP_ID, x + LANE_ID) = input_operator(l_lhs[y * stride_ly + x * stride_lx]);
          }
        }
      }
    }
    if (tail_w < Ntile) {
      if (is_right_row_major) {
        for (int y = 0; y < Ktile; y += Warps) {
          for (int x = 0; x < Ntile; x += Lanes) {
            if (x >= lane_tail_w) {
              RHS(y + WARP_ID, x + LANE_ID) = 0;
            } else {
              RHS(y + WARP_ID, x + LANE_ID) = input_operator(l_rhs[y * stride_ry + x]);
            }
          }
        }
      } else {
        for (int y = 0; y < Ktile; y += Warps) {
          for (int x = 0; x < Ntile; x += Lanes) {
            if (x >= lane_tail_w) {
              RHS(y + WARP_ID, x + LANE_ID) = 0;
            } else {
              RHS(y + WARP_ID, x + LANE_ID) = input_operator(l_rhs[y * stride_ry + x * stride_rx]);
            }
          }
        }
      }
    } else {
      if (is_right_row_major) {
        for (int y = 0; y < Ktile; y += Warps) {
          for (int x = 0; x < Ntile; x += Lanes) {
            RHS(y + WARP_ID, x + LANE_ID) = input_operator(l_rhs[y * stride_ry + x]);
          }
        }
      } else {
        for (int y = 0; y < Ktile; y += Warps) {
          for (int x = 0; x < Ntile; x += Lanes) {
            RHS(y + WARP_ID, x + LANE_ID) = input_operator(l_rhs[y * stride_ry + x * stride_rx]);
          }
        }
      }
    }
    __syncthreads();
    for (int kk = 0; kk < Ktile; ++kk) {
      for (int y = 0; y < AccumY; ++y) {
        for (int x = 0; x < AccumX; ++x) {
          v_accum[y][x] += to_float(LHS(y * Warps + WARP_ID, kk) * RHS(kk, x * Lanes + LANE_ID));
        }
      }
    }
    k -= Ktile;
    lhs += Ktile * stride_lx;
    rhs += Ktile * stride_ry;
    __syncthreads();
  }

  // trash case. don't hit this.
  while (k > 0) {
    for (int y = 0; y < AccumY; ++y) {
      for (int x = 0; x < AccumX; ++x) {
        int warp_y = y * Warps + WARP_ID;
        int lane_x = x * Lanes + LANE_ID;
        if (warp_y < tail_h && lane_x < tail_w) {
          v_accum[y][x] += to_float(input_operator(lhs[(y * Warps + WARP_ID) * stride_ly]) *
                                    input_operator(rhs[(x * Lanes + LANE_ID) * stride_rx]));
        }
      }
    }

    k -= 1;
    lhs += stride_lx;
    rhs += stride_ry;
  }

  // output
  for (int y = 0; y < AccumY; ++y) {
    for (int x = 0; x < AccumX; ++x) {
      int warp_y = y * Warps + WARP_ID;
      int lane_x = x * Lanes + LANE_ID;
      int abs_y = warp_y + by * Mtile;
      int abs_x = lane_x + bx * Ntile;
      if (warp_y < tail_h && lane_x < tail_w) {
        output[abs_y * stride_oy + abs_x * stride_ox] =
            mask_operator(abs_x, abs_y) ? -CUDART_INF_F
                                        : to_float(output_operator(v_accum[y][x], output, abs_x,
                                                                   abs_y, stride_ox, stride_oy));
      }
    }
  }
}

#define MATMUL_GENERIC_MASK_OP(op)                               \
  switch (mask) {                                                \
    case MatmulMaskOp::NONE: {                                   \
      op(MATMUL_COMMON_ARGS(), load_op, store_op, NoMask());     \
      break;                                                     \
    }                                                            \
    case MatmulMaskOp::CAUSAL: {                                 \
      op(MATMUL_COMMON_ARGS(), load_op, store_op, CausalMask()); \
      break;                                                     \
    }                                                            \
    default:                                                     \
      NYI();                                                     \
  }

#define MATMUL_GENERIC_LOAD_OP(op)                     \
  switch (load) {                                      \
    case MatmulLoadOp::IDENTITY: {                     \
      Identity<T> load_op;                             \
      MATMUL_GENERIC_MASK_OP(op)                       \
      break;                                           \
    }                                                  \
    case MatmulLoadOp::SCALE: {                        \
      auto load_op = Scale<T, Identity<T>>(scale, {}); \
      MATMUL_GENERIC_MASK_OP(op)                       \
      break;                                           \
    }                                                  \
    default:                                           \
      NYI();                                           \
      break;                                           \
  }

// TODO: half2float here breaks for other T
#define MATMUL_GENERIC_STORE_OP(op)                                              \
  switch (store) {                                                               \
    case MatmulStoreOp::IDENTITY: {                                              \
      Identity<T> store_op;                                                      \
      MATMUL_GENERIC_LOAD_OP(op)                                                 \
      break;                                                                     \
    }                                                                            \
    case MatmulStoreOp::BETA_GELU_BIAS: {                                        \
      using bias_t = Bias<float, T, HalfToFloat<>>;                              \
      using gelu_t = Gelu<float, T, bias_t>;                                     \
      using beta_t = Beta<float, T, gelu_t>;                                     \
      beta_t store_op{beta, gelu_t(bias_t(bias, HalfToFloat<>(Identity<T>())))}; \
      MATMUL_GENERIC_LOAD_OP(op)                                                 \
      break;                                                                     \
    }                                                                            \
    case MatmulStoreOp::BETA_BIAS: {                                             \
      using bias_t = Bias<float, T, HalfToFloat<>>;                              \
      using beta_t = Beta<float, T, bias_t>;                                     \
      beta_t store_op{beta, bias_t(bias, HalfToFloat<>(Identity<T>()))};         \
      MATMUL_GENERIC_LOAD_OP(op)                                                 \
      break;                                                                     \
    }                                                                            \
    case MatmulStoreOp::SCALE: {                                                 \
      auto store_op = Scale<T, Identity<T>>(scale, {});                          \
      MATMUL_GENERIC_LOAD_OP(op)                                                 \
      break;                                                                     \
    }                                                                            \
    case MatmulStoreOp::ADD: {                                                   \
      auto store_op = Add<T, T, Identity<T>>({});                                \
      MATMUL_GENERIC_LOAD_OP(op)                                                 \
      break;                                                                     \
    }                                                                            \
    default:                                                                     \
      NYI();                                                                     \
  }

extern "C" __global__ void matmul_f16(MATMUL_COMMON_PARAMS(__half),
                                      __half const* const bias,
                                      float const beta = 0.0f,
                                      float const scale = 1.0f,
                                      MatmulStoreOp store = MatmulStoreOp::IDENTITY,
                                      MatmulLoadOp load = MatmulLoadOp::IDENTITY,
                                      MatmulMaskOp mask = MatmulMaskOp::NONE) {
  using T = __half;
  MATMUL_GENERIC_STORE_OP(matmul)
}

extern "C" __global__ void matmul_f16_fast(MATMUL_COMMON_PARAMS(__half),
                                           __half const* const bias,
                                           float const beta = 0.0f,
                                           float const scale = 1.0f,
                                           MatmulStoreOp store = MatmulStoreOp::IDENTITY,
                                           MatmulLoadOp load = MatmulLoadOp::IDENTITY,
                                           MatmulMaskOp mask = MatmulMaskOp::NONE) {
  using T = __half;
  MATMUL_GENERIC_STORE_OP(matmul_fast)
}

extern "C" __global__ void matmul_f16_raw(MATMUL_COMMON_PARAMS(__half)) {
  matmul_fast(MATMUL_COMMON_ARGS(), {}, {}, {});
}
