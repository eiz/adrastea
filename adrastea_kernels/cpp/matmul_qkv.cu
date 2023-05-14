#include <cuda_fp16.h>

// lhs: (n_heads, seq_len_new, seq_len)
// rhs: (seq_len, dim)
// output: (seq_len_new, dim)
// logically we are viewing rhs as (seq_len, n_heads, head_dim) and then swizzling to
// (n_heads, seq_len, head_dim)
// then (n_heads, seq_len_new, seq_len) * (n_heads, seq_len, head_dim) ==>
// transpose (n_heads, seq_len_new, head_dim) ==>
// view (seq_len_new, n_heads, head_dim) ==>
// (seq_len_new, dim)
// but we're going to do all that without actually doing the transposes!
extern "C" __global__ void matmul_qkv(__half* output,
                                      __half* lhs,
                                      __half* rhs,
                                      int seq_len_new,
                                      int seq_len,
                                      int dim,
                                      int n_heads) {
  // TODO: write a tiled version of this.
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int head_dim = dim / n_heads;
  int head = c / head_dim;
  if (r < seq_len_new && c < dim) {
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
      sum += __half2float(lhs[head * seq_len_new * seq_len + r * seq_len + i]) *
             __half2float(rhs[i * dim + c]);
    }
    output[r * dim + c] = __float2half(sum);
  }
}
