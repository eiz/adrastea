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
#else  // __CUDACC__
#include <hip/hip_fp16.h>
#include <hip/hip_math_constants.h>
#include <hip/hip_runtime.h>
#define THREAD_IDX_X hipThreadIdx_x
#define THREAD_IDX_Y hipThreadIdx_y
#define THREAD_IDX_Z hipThreadIdx_z
#define BLOCK_IDX_X hipBlockIdx_x
#define BLOCK_IDX_Y hipBlockIdx_y
#define BLOCK_IDX_Z hipBlockIdx_z
#define BLOCK_DIM_X hipBlockDim_x
#define BLOCK_DIM_Y hipBlockDim_y
#define BLOCK_DIM_Z hipBlockDim_z
#define CUDART_INF_F HIP_INF_F
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)
#endif  // !__CUDACC__
