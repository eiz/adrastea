import json
import os

import matmul
import square

from triton.compiler import compile_artifacts


def compile_one(fn, name, **kwargs):
    fn2, so_path, metadata, asm = compile_artifacts(fn, **kwargs)
    cc = str(kwargs.get("cc"))
    os.makedirs(f"arch/cuda/{cc}", exist_ok=True)
    print(f"arch/cuda/{cc}/{name}.cubin")
    with open(f"arch/cuda/{cc}/{name}.ptx", "w") as f:
        f.write(asm["ptx"])
    with open(f"arch/cuda/{cc}/{name}.cubin", "wb") as f:
        f.write(asm["cubin"])
    with open(f"arch/cuda/{cc}/{name}.json", "w") as f:
        f.write(json.dumps(metadata, indent=2))


compile_one(
    square.square,
    "square_fp32_16x16",
    signature="*fp32,*fp32,i32,i32",
    constants={"BLOCK_SIZE_X": 16, "BLOCK_SIZE_Y": 16},
    num_warps=8,
    cc=80,
)

for config in matmul.matmul_kernel.configs:
    cd = config.kwargs
    compile_one(
        matmul.matmul_kernel.fn,
        f"matmul_fp16_{cd['BLOCK_SIZE_M']}x{cd['BLOCK_SIZE_N']}x{cd['BLOCK_SIZE_K']}_{cd['GROUP_SIZE_M']}_{config.num_stages}_{config.num_warps}",
        signature="*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,i32",
        constants={"ACTIVATION": "", **config.kwargs},
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        cc=80,
    )
