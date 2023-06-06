#include "compat.h"

#ifdef __gfx1100__
#define NOP_LOOP(n)                                    \
  extern "C" void __global__ nop_loop_##n(int times) { \
    for (int i = 0; i < times; ++i) {                  \
      asm volatile("s_nop " #n);                       \
    }                                                  \
  }

NOP_LOOP(1)
NOP_LOOP(2)
NOP_LOOP(3)
NOP_LOOP(4)
NOP_LOOP(5)
NOP_LOOP(6)
NOP_LOOP(7)
NOP_LOOP(8)
NOP_LOOP(9)
NOP_LOOP(10)
NOP_LOOP(11)
NOP_LOOP(12)
NOP_LOOP(13)
NOP_LOOP(14)
NOP_LOOP(15)

extern "C" void __global__ wmma_loop(int times) {
  for (int i = 0; i < times; ++i) {
    asm volatile("v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23]");
  }
}
#endif

extern "C" void __global__ empty_kernel(int derp) {
  //
}
