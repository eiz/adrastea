#include "compat.h"

struct stack_frame {
  int dim_idx;
  int idx;
};

extern "C" __global__ void fp32_to_fp16(TensorViewF16 output, TensorViewF32 input) {
  int ox = BLOCK_IDX_X * BLOCK_DIM_X + THREAD_IDX_X;
  int oy = BLOCK_IDX_Y * BLOCK_DIM_Y + THREAD_IDX_Y;
  stack_frame dim_stack[7];
  stack_frame* bos = &dim_stack[7];
  stack_frame* tos = &dim_stack[6];
  tos->dim_idx = output.dims - 1;
  tos->idx = 0;
#define POP()        \
  do {               \
    tos++;           \
    if (tos < bos) { \
      tos->idx++;    \
    }                \
  } while (0)
  while (tos < bos) {
    if (tos->dim_idx < 2) {
      TensorViewF16 output_slice = output;
      TensorViewF32 input_slice = input;
      for (stack_frame* cur = tos + 1; cur < bos; ++cur) {
        output_slice.ptr += cur->idx * output.strides[cur->dim_idx];
        input_slice.ptr += cur->idx * input.strides[cur->dim_idx];
      }
      output_slice.dims = 2;
      input_slice.dims = 2;
      if (ox < output.width() && oy < max(output.height(), 1)) {
        output_slice(oy, ox) = input_slice(oy, ox);
      }
      POP();
    } else {
      if (tos->idx >= output.shape[tos->dim_idx]) {
        POP();
        continue;
      }
      tos--;
      tos->dim_idx = tos[1].dim_idx - 1;
      tos->idx = 0;
    }
  }
}