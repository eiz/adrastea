#include "compat.h"

// https://stackoverflow.com/a/51549250
// technically this does not handle nan properly but those dont exist right
// actually my sums are always positive so you dont even need the negative case
__device__ __forceinline__ float atomic_max_float(float* addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                     : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

// output[0] = mean squared error, output[1] = max error, output[2] = mean absolute error
template <typename T>
__device__ void error_stats(TensorViewF32 output, TensorView<T> lhs, TensorView<T> rhs) {
  __shared__ float s_sum_sq_reduced[32];
  __shared__ float s_max_reduced[32];
  __shared__ float s_sum_reduced[32];
  float sum_sq = 0;
  float max = 0;
  float sum = 0;
  int tid = threadIdx.x;
  int lane_id = tid % 32;
  int warp_id = tid / 32;
  int length = rhs.width();
  bool warp_leader = lane_id == 0;
  int block_size = length / gridDim.x;
  int block_start = block_size * blockIdx.x;
  int block_end = block_size * (blockIdx.x + 1);
  for (int i = block_start + tid; i < block_end && i < length; i += blockDim.x) {
    float diff = float(lhs(i)) - float(rhs(i));
    sum_sq += diff * diff;
    max = fmax(max, fabs(diff));
    sum += fabs(diff);
  }
  __syncthreads();
  for (int i = 16; i > 0; i /= 2) {
    sum_sq += __shfl_xor_sync(~0, sum_sq, i);
    max = fmax(max, __shfl_xor_sync(~0, max, i));
    sum += __shfl_xor_sync(~0, sum, i);
  }
  if (warp_leader) {
    s_sum_sq_reduced[warp_id] = sum_sq;
    s_max_reduced[warp_id] = max;
    s_sum_reduced[warp_id] = sum;
  }
  __syncthreads();
  if (warp_id == 0) {
    sum_sq = lane_id < (blockDim.x / 32) ? s_sum_sq_reduced[lane_id] : 0;
    max = lane_id < (blockDim.x / 32) ? s_max_reduced[lane_id] : 0;
    sum = lane_id < (blockDim.x / 32) ? s_sum_reduced[lane_id] : 0;
    for (int i = 16; i > 0; i /= 2) {
      sum_sq += __shfl_xor_sync(~0, sum_sq, i);
      max = fmax(max, __shfl_xor_sync(~0, max, i));
      sum += __shfl_xor_sync(~0, sum, i);
    }
    if (warp_leader) {
      atomicAdd(&output(0), sum_sq / length);
      atomic_max_float(&output(1), max);
      atomicAdd(&output(2), sum / length);
    }
  }
}

extern "C" __global__ void error_stats_f16(TensorViewF32 output,
                                           TensorViewF16 lhs,
                                           TensorViewF16 rhs) {
  error_stats(output, lhs, rhs);
}

extern "C" __global__ void error_stats_f32(TensorViewF32 output,
                                           TensorViewF32 lhs,
                                           TensorViewF32 rhs) {
  error_stats(output, lhs, rhs);
}