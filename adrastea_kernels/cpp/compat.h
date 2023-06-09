#pragma once

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <math_constants.h>
#define THREAD_IDX_X threadIdx.x
#define THREAD_IDX_Y threadIdx.y
#define THREAD_IDX_Z threadIdx.z
#define BLOCK_IDX_X blockIdx.x
#define BLOCK_IDX_Y blockIdx.y
#define BLOCK_IDX_Z blockIdx.z
#define BLOCK_DIM_X blockDim.x
#define BLOCK_DIM_Y blockDim.y
#define BLOCK_DIM_Z blockDim.z
#define GRID_DIM_X gridDim.x
#define GRID_DIM_Y gridDim.y
#define GRID_DIM_Z gridDim.z
#else  // __CUDACC__
#include <hip/hip_fp16.h>
#include <hip/hip_math_constants.h>
#include <hip/hip_runtime.h>
#define THREAD_IDX_X __builtin_amdgcn_workitem_id_x()
#define THREAD_IDX_Y __builtin_amdgcn_workitem_id_y()
#define THREAD_IDX_Z __builtin_amdgcn_workitem_id_z()
#define BLOCK_IDX_X __builtin_amdgcn_workgroup_id_x()
#define BLOCK_IDX_Y __builtin_amdgcn_workgroup_id_y()
#define BLOCK_IDX_Z __builtin_amdgcn_workgroup_id_z()
#define BLOCK_DIM_X hipBlockDim_x
#define BLOCK_DIM_Y hipBlockDim_y
#define BLOCK_DIM_Z hipBlockDim_z
#define GRID_DIM_X hipGridDim_x
#define GRID_DIM_Y hipGridDim_y
#define GRID_DIM_Z hipGridDim_z
#define CUDART_INF_F HIP_INF_F
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)
#endif  // !__CUDACC__

template <typename T>
struct TensorView {
  T* ptr;
  uint32_t dims;
  uint32_t shape[7];
  uint32_t strides[7];

  struct StackFrame {
    int dim_idx;
    int idx;
  };

  // traditional NCHW dimension names
  __device__ __forceinline__ uint32_t width() { return shape[0]; }
  __device__ __forceinline__ uint32_t height() { return shape[1]; }
  __device__ __forceinline__ uint32_t channels() { return shape[2]; }
  __device__ __forceinline__ uint32_t batches() { return shape[3]; }

  __device__ __forceinline__ T& operator()(uint32_t x) { return ptr[x * strides[0]]; }
  __device__ __forceinline__ T& operator()(uint32_t y, uint32_t x) {
    return ptr[y * strides[1] + x * strides[0]];
  }
  __device__ __forceinline__ T& operator()(uint32_t c, uint32_t y, uint32_t x) {
    return ptr[c * strides[2] + y * strides[1] + x * strides[0]];
  }
  __device__ __forceinline__ T& operator()(uint32_t n, uint32_t c, uint32_t y, uint32_t x) {
    return ptr[n * strides[3] + c * strides[2] + y * strides[1] + x * strides[0]];
  }

  __device__ __forceinline__ TensorView<T> slice(StackFrame* tos, StackFrame* bos) {
    auto result = *this;
    for (StackFrame* cur = tos + 1; cur < bos; cur++) {
      result.ptr += cur->idx * result.strides[cur->dim_idx];
      result.dims--;
    }
    return result;
  }

  template <typename Fn>
  __device__ __forceinline__ void iter_dims(int atom_dim, Fn fn) {
    StackFrame dim_stack[7];
    StackFrame* bos = &dim_stack[7];
    StackFrame* tos = &dim_stack[6];
    tos->dim_idx = dims - 1;
    tos->idx = 0;
#define POP()        \
  do {               \
    tos++;           \
    if (tos < bos) { \
      tos->idx++;    \
    }                \
  } while (0)
    while (tos < bos) {
      if (tos->dim_idx <= atom_dim) {
        fn(tos, bos);
        POP();
      } else {
        if (tos->idx >= shape[tos->dim_idx]) {
          POP();
          continue;
        }
        tos--;
        tos->dim_idx = tos[1].dim_idx - 1;
        tos->idx = 0;
      }
    }
#undef POP
  }
};

using TensorViewU8 = TensorView<uint8_t>;
using TensorViewF16 = TensorView<__half>;
using TensorViewF32 = TensorView<float>;
