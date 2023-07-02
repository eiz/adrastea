#include "compat.h"

// all normalizations are performed in f32 and converted to the desired output format

// row-wise rms normalization
// 1 block per row, x = row, 8 warps per block
extern "C" __global__ void rms_norm(__half* output,
                                    __half* input,
                                    __half* weights,
                                    int length_x,
                                    int length_y,
                                    int stride_ox,
                                    int stride_oy,
                                    int stride_ix,
                                    int stride_iy,
                                    float eps) {
  int row = BLOCK_IDX_X;
  int tid = THREAD_IDX_X;
  int in_row_idx = row * stride_iy;
  int out_row_idx = row * stride_oy;
  int warp_id = tid / 32;
  bool warp_leader = (tid % 32) == 0;
  __shared__ float s_rms_inv;
  __shared__ float s_warp_reduced[8];
  float sum_val = 0.0f;
  // sum_sq: thread reduction
  for (int i = tid; i < length_x; i += BLOCK_DIM_X) {
    float val = __half2float(input[in_row_idx + i * stride_ix]);
    sum_val += val * val;
  }
  __syncthreads();
  // sum_sq: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_xor_sync(~0, sum_val, offset);
    sum_val += other_val;
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = sum_val;
  }
  // sum_sq: block reduction
  __syncthreads();
  if (warp_id == 0) {
    sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
    for (int offset = 4; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, sum_val, offset);
      sum_val += other_val;
    }
    if (warp_leader) {
      s_rms_inv = rsqrt((sum_val / length_x) + eps);
    }
  }
  __syncthreads();
  float rms_inv = s_rms_inv;
  for (int i = tid; i < length_x; i += BLOCK_DIM_X) {
    float val = __half2float(input[in_row_idx + i * stride_ix]);
    output[out_row_idx + i * stride_ix] = weights[i] * __float2half(val * rms_inv);
  }
}

// row-wise layer normalization
// 1 block per row, x = row, 8 warps per block
// TODO strides kill the performance, refuxor later
extern "C" __global__ void layer_norm(__half* output,
                                      __half* input,
                                      __half* weights,
                                      __half* bias,
                                      int length_x,
                                      int length_y,
                                      int stride_ox,
                                      int stride_oy,
                                      int stride_ix,
                                      int stride_iy,
                                      float eps) {
  int row = BLOCK_IDX_X;
  int tid = THREAD_IDX_X;
  int in_row_idx = row * stride_iy;
  int out_row_idx = row * stride_oy;
  int warp_id = tid / 32;
  bool warp_leader = (tid % 32) == 0;
  __shared__ float s_mean;
  __shared__ float s_stddev;
  __shared__ float s_warp_reduced[8];
  float sum_val = 0.0f;
  // sum: thread reduction
  for (int i = tid; i < length_x; i += BLOCK_DIM_X) {
    float val = __half2float(input[in_row_idx + i * stride_ix]);
    sum_val += val;
  }
  __syncthreads();
  // sum: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_xor_sync(~0, sum_val, offset);
    sum_val += other_val;
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = sum_val;
  }
  // sum: block reduction
  __syncthreads();
  if (warp_id == 0) {
    sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
    for (int offset = 4; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, sum_val, offset);
      sum_val += other_val;
    }
    if (warp_leader) {
      s_mean = sum_val / length_x;
    }
  }
  __syncthreads();
  sum_val = 0.0f;
  // mean diff: thread reduction
  for (int i = tid; i < length_x; i += BLOCK_DIM_X) {
    float val = __half2float(input[in_row_idx + i * stride_ix]);
    sum_val += (val - s_mean) * (val - s_mean);
  }
  __syncthreads();
  // mean diff: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_xor_sync(~0, sum_val, offset);
    sum_val += other_val;
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = sum_val;
  }
  // mean diff: block reduction
  __syncthreads();
  if (warp_id == 0) {
    sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
    for (int offset = 4; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, sum_val, offset);
      sum_val += other_val;
    }
    if (warp_leader) {
      s_stddev = sqrt(sum_val / length_x + eps);
    }
  }
  __syncthreads();
  for (int i = tid; i < length_x; i += BLOCK_DIM_X) {
    output[out_row_idx + i * stride_ox] =
        (__half2float(input[in_row_idx + i * stride_ix]) - s_mean) / s_stddev *
            __half2float(weights[i]) +
        __half2float(bias[i]);
  }
}

// row-wise softmax with optional temperature
// input = (n_heads, seq_len, seq_len)
// output = (n_heads, seq_len, seq_len)
extern "C" __global__ void softmax_rows(__half* output,
                                        __half* input,
                                        int h,
                                        int w,
                                        float temp = 1.0) {
  int row = BLOCK_IDX_X;
  int head = BLOCK_IDX_Y;
  int tid = THREAD_IDX_X;
  int row_idx = head * h * w + row * w;
  bool warp_leader = tid % 32 == 0;
  int warp_id = tid / 32;
  __shared__ float s_max_val, s_sum_exp;
  __shared__ float s_warp_reduced[8];
  // max: thread reduction
  float max_val = -CUDART_INF_F;
  for (int i = tid; i < w; i += BLOCK_DIM_X) {
    max_val = fmaxf(max_val, __half2float(input[row_idx + i]) / temp);
  }
  __syncthreads();
  // max: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_xor_sync(~0, max_val, offset);
    max_val = fmaxf(max_val, other_val);
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = max_val;
  }
  // max: block reduction
  __syncthreads();
  if (warp_id == 0) {
    max_val = (tid < 8) ? s_warp_reduced[tid] : -CUDART_INF_F;
    for (int offset = 4; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, max_val, offset);
      max_val = fmaxf(max_val, other_val);
    }
    if (warp_leader) {
      s_max_val = max_val;
    }
  }
  __syncthreads();
  float sum_val = 0.0f;
  // expsum: thread reduction
  for (int i = tid; i < w; i += BLOCK_DIM_X) {
    float val = __half2float(input[row_idx + i]) / temp;
    sum_val += expf(val - s_max_val);
  }
  __syncthreads();
  // expsum: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_val += __shfl_xor_sync(~0, sum_val, offset);
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = sum_val;
  }
  // expsum: block reduction
  __syncthreads();
  if (warp_id == 0) {
    sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
    for (int offset = 4; offset > 0; offset /= 2) {
      sum_val += __shfl_xor_sync(~0, sum_val, offset);
    }
    if (warp_leader) {
      s_sum_exp = sum_val;
    }
  }
  __syncthreads();
  for (int i = tid; i < w; i += BLOCK_DIM_X) {
    float val = __half2float(input[row_idx + i]) / temp;
    output[row_idx + i] = __float2half(expf(val - s_max_val) / s_sum_exp);
  }
}

// row-wise euclidean vector norm
template <typename T>
__device__ void l2_norm(TensorView<T> output, TensorView<T> input) {
  using Frame = typename TensorView<T>::StackFrame;
  output.iter_dims(1, [&output, &input](Frame* tos, Frame* bos) {
    auto out_view = output.slice(tos, bos);
    auto in_view = input.slice(tos, bos);
    int row = BLOCK_IDX_X;
    int tid = THREAD_IDX_X;
    int warp_id = tid / 32;
    bool warp_leader = (tid % 32) == 0;
    __shared__ float s_norm;
    __shared__ float s_warp_reduced[8];
    float sum_val = 0.0f;
    // sum_sq: thread reduction
    for (int i = tid; i < out_view.width(); i += BLOCK_DIM_X) {
      float val = float(in_view(row, i));
      sum_val += val * val;
    }
    __syncthreads();
    // sum_sq: warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, sum_val, offset);
      sum_val += other_val;
    }
    if (warp_leader) {
      s_warp_reduced[warp_id] = sum_val;
    }
    // sum_sq: block reduction
    __syncthreads();
    if (warp_id == 0) {
      sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
      for (int offset = 4; offset > 0; offset /= 2) {
        float other_val = __shfl_xor_sync(~0, sum_val, offset);
        sum_val += other_val;
      }
      if (warp_leader) {
        s_norm = rsqrt(sum_val + 1.0e-5);
      }
    }
    __syncthreads();
    float norm = s_norm;
    for (int i = tid; i < out_view.width(); i += BLOCK_DIM_X) {
      output(row, i) = T(float(in_view(row, i)) * norm);
    }
  });
}

extern "C" __global__ void l2_norm_f16(TensorViewF16 output, TensorViewF16 input) {
  l2_norm(output, input);
}

extern "C" __global__ void l2_norm_f32(TensorViewF32 output, TensorViewF32 input) {
  l2_norm(output, input);
}