#include "compat.h"

extern "C" __global__ void fp32_to_fp16(TensorViewF16 output, TensorViewF16 input) {
  int ox = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int oy = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  if (ox < output.width() && oy < output.height()) {
    output(oy, ox) = input(oy, ox);
  }
}