#include "compat.h"

extern "C" __global__ void embed_uint8(int n_dim,
                                       int n_ids,
                                       short* in_ids,
                                       __half* out_embeddings,
                                       uint8_t* embeddings_table,
                                       half* scales,
                                       int block_size) {
  int tid = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int block_count = n_dim / block_size;
  if (tid < n_ids) {
    int id = in_ids[tid];
    for (int i = 0; i < n_dim; i++) {
      out_embeddings[tid * n_dim + i] = (float(embeddings_table[id * n_dim + i]) - 127.5f) *
                                        __half2float(scales[id * block_count + i / block_size]);
    }
  }
}
