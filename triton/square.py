import json
import os

import torch
import triton
import triton.language as tl


@triton.jit
def square(
    out_ptr,
    in_ptr,
    width,
    height,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    px, py = tl.program_id(0), tl.program_id(1)
    range_x, range_y = tl.arange(0, BLOCK_SIZE_X), tl.arange(0, BLOCK_SIZE_Y)
    block_xs, block_ys = px * BLOCK_SIZE_X + range_x, py * BLOCK_SIZE_Y + range_y

    offsets = block_ys[:, None] * width + block_xs[None, :]
    mask = (block_ys < height)[:, None] & (block_xs < width)[None, :]
    in_mem = tl.load(in_ptr + offsets, mask=mask)

    tl.store(out_ptr + offsets, in_mem * in_mem, mask=mask)
