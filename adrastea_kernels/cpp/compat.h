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