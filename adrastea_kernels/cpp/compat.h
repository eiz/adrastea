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
