#include "compat.h"

__global__ void quantize_absmax_uint8(uint8_t* output,
                                      half* scales,
                                      half const* __restrict__ input,
                                      int M,
                                      int N,
                                      int K) {
  __shared__ float s_warp_reduced[32];
  __shared__ half s_absmax;
  float absmax_val = 0;
  int row = blockIdx.y * blockDim.y;
  int tid = threadIdx.x;
  int lane_id = tid % 32;
  int warp_id = tid / 32;
  bool warp_leader = lane_id == 0;
  int block_count = N / K;
  for (int i = blockIdx.x * K + tid; i < (blockIdx.x + 1) * K; i += blockDim.x) {
    float val = __float2half(input[row * N + i]);
    absmax_val = fmax(absmax_val, fabs(val));
  }
  __syncthreads();
  for (int i = 16; i > 0; i /= 2) {
    absmax_val = fmax(absmax_val, __shfl_xor_sync(~0, absmax_val, i));
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = absmax_val;
  }
  __syncthreads();
  if (warp_id == 0) {
    absmax_val = lane_id < (blockDim.x / 32) ? s_warp_reduced[lane_id] : 0;
    for (int i = 16; i > 0; i /= 2) {
      absmax_val = fmax(absmax_val, __shfl_xor_sync(~0, absmax_val, i));
    }
    if (warp_leader) {
      s_absmax = absmax_val / 127.5f;
    }
  }
  __syncthreads();
  if (tid == 0) {
    scales[row * block_count + blockIdx.x] = s_absmax;
  }
  for (int i = blockIdx.x * K + tid; i < (blockIdx.x + 1) * K; i += blockDim.x) {
    output[row * N + i] = static_cast<uint8_t>(
        round(127.5f + __half2float(input[row * N + i]) / __half2float(s_absmax)));
  }
}

__global__ void dequantize_absmax_uint8(half* output,
                                        half const* __restrict__ scales,
                                        uint8_t const* __restrict__ input,
                                        int M,
                                        int N,
                                        int K) {
  int row = blockIdx.y;
  int tid = threadIdx.x;
  int block_count = N / K;
  for (int i = blockIdx.x * K + tid; i < (blockIdx.x + 1) * K; i += blockDim.x) {
    output[row * N + i] = __float2half((float(input[row * N + i]) - 127.5f) *
                                       __half2float(scales[row * block_count + blockIdx.x]));
  }
}