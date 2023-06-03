#include "compat.h"

// TODO: these embed kernels are objectively terrible
extern "C" __global__ void embed(__half* out_embeddings,
                                 int* in_ids,
                                 int n_dim,
                                 int n_ids,
                                 __half* embeddings_table) {
  int tid = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  if (tid < n_ids) {
    int id = in_ids[tid];
    for (int i = 0; i < n_dim; i++) {
      out_embeddings[tid * n_dim + i] = embeddings_table[id * n_dim + i];
    }
  }
}
