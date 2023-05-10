import json
import os

import torch
import triton
import triton.language as tl

from triton.compiler import compile_artifacts


@triton.jit
def square(
    out_ptr,
    in_ptr,
    width,
    height,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    px = tl.program_id(0)
    py = tl.program_id(1)
    range_x = tl.arange(0, BLOCK_SIZE_X)
    range_y = tl.arange(0, BLOCK_SIZE_Y)
    block_xs = px * BLOCK_SIZE_X + range_x
    block_ys = py * BLOCK_SIZE_Y + range_y
    offsets = block_ys[:, None] * width + block_xs[None, :]
    mask = (block_ys < height)[:, None] & (block_xs < width)[None, :]
    in_mem = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, in_mem * in_mem, mask=mask)


def compile_one(fn, name, **kwargs):
    fn2, so_path, metadata, asm = compile_artifacts(fn, **kwargs)
    cc = str(kwargs.get("cc"))
    os.makedirs(f"arch/cuda/{cc}", exist_ok=True)
    with open(f"arch/cuda/{cc}/{name}.ptx", "w") as f:
        f.write(asm["ptx"])
    with open(f"arch/cuda/{cc}/{name}.cubin", "wb") as f:
        f.write(asm["cubin"])
    with open(f"arch/cuda/{cc}/{name}.json", "w") as f:
        f.write(json.dumps(metadata, indent=2))


compile_one(
    square,
    "square_fp32_16x16",
    signature="*fp32,*fp32,i32,i32",
    constants={"BLOCK_SIZE_X": 16, "BLOCK_SIZE_Y": 16},
    num_warps=8,
    cc=80,
)
