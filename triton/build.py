from collections import namedtuple
import json
import os

import matmul
import square

from triton.compiler import compile_artifacts


def compile_one(fn, name, **kwargs):
    constants = kwargs.get("constants", {})
    arg_idx = {f"{arg}": i for i, arg in enumerate(fn.arg_names)}
    constants = {
        arg_idx[constant_name]: value for constant_name, value in constants.items()
    }
    fn2, so_path, metadata, asm = compile_artifacts(
        fn, **{**kwargs, "constants": constants}
    )
    cc = str(kwargs.get("cc"))
    os.makedirs(f"arch/cuda/{cc}", exist_ok=True)
    print(f"arch/cuda/{cc}/{name}.cubin")
    with open(f"arch/cuda/{cc}/{name}.ptx", "w") as f:
        f.write(asm["ptx"])
    with open(f"arch/cuda/{cc}/{name}.cubin", "wb") as f:
        f.write(asm["cubin"])
    with open(f"arch/cuda/{cc}/{name}.json", "w") as f:
        f.write(json.dumps(metadata, indent=2))


for cc in [80, 89]:
    compile_one(
        square.square,
        "square_fp32_16x16",
        signature="*fp32,*fp32,i32,i32",
        constants={"BLOCK_SIZE_X": 16, "BLOCK_SIZE_Y": 16},
        num_warps=8,
        cc=cc,
    )

    for config in matmul.matmul_kernel.configs:
        cd = config.kwargs
        compile_one(
            matmul.matmul_kernel.fn,
            f"matmul_fp16_{cd['BLOCK_SIZE_M']}x{cd['BLOCK_SIZE_N']}x{cd['BLOCK_SIZE_K']}_{cd['GROUP_SIZE_M']}_{config.num_stages}_{config.num_warps}",
            signature="*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,i32",
            constants={
                "ACTIVATION": "",
                **config.kwargs,
                "stride_ak": 1,
                "stride_bn": 1,
                "stride_cn": 1,
            },
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            cc=cc,
            configs=(
                namedtuple("instance_descriptor", ["divisible_by_16", "equal_to_1"])(
                    (
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        8,
                        10,
                        12,
                        13,
                        14,
                    ),
                    (7, 9, 11),
                ),
            ),
        )
