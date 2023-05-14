#include <cuda_fp16.h>

// TODO: these embed kernels are objectively terrible
extern "C" __global__ void embed(int n_dim,
                                 int n_ids,
                                 short* in_ids,
                                 __half* out_embeddings,
                                 __half* embeddings_table) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_ids) {
    int id = in_ids[tid];
    for (int i = 0; i < n_dim; i++) {
      out_embeddings[tid * n_dim + i] = embeddings_table[id * n_dim + i];
    }
  }
}
