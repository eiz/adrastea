#include "compat.h"

// q * k^T
// output = (n_heads, seq_len_new, seq_len)
// lhs = (seq_len_new, dim)
// rhs = (seq_len, dim)
// we have to do the swizzle to regroup by heads here
// basically we have the above as inputs but we want to access them like
// lhs = (n_heads, seq_len_new, head_dim)
// rhs = (n_heads, head_dim, seq_len)
// grid x/y are row/col indices for the output, z is the head index
extern "C" __global__ void matmul_qk(__half* output,
                                     __half* lhs,
                                     __half* rhs,
                                     int seq_len_new,
                                     int seq_len,
                                     int dim,
                                     int n_heads,
                                     int start_pos) {
  // TODO: write a tiled kernel for this. only for testing accuracy.
  // probably write a cuBLAS path too which will need to materialize the
  // transposes.
  int c = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int r = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  int head = blockIdx.z;
  int head_dim = dim / n_heads;
  bool masked = c > (r + start_pos);
  if (r < seq_len_new && c < seq_len) {
    if (masked) {
      output[head * seq_len_new * seq_len + r * seq_len_new + c] = -CUDART_INF_F;
    } else {
      float sum = 0;
      for (int i = 0; i < head_dim; i++) {
        sum += __half2float(lhs[r * dim + head * head_dim + i]) *
               __half2float(rhs[c * dim + head * head_dim + i]);
      }
      output[head * seq_len_new * seq_len + r * seq_len_new + c] = sum / sqrt(float(head_dim));
    }
  }
}
