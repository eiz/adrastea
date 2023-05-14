use std::{fs, path::PathBuf, process::Command};

fn main() {
    let kernels = &[
        "embed",
        "embed_uint8",
        "matmul_nt_fp16u8",
        "matmul_nt_wmma_16x128x256_fp16u8",
        "matmul_nt_wmma_16x128x256",
        "matmul_nt_wmma_128x64x64",
        "matmul_nt",
        "matmul_qk",
        "matmul_qkv",
        "rms_norm",
        "rotary",
        "silu",
        "softmax_rows",
        "square_fp32_16x16",
    ];
    let arches = &["sm_80", "sm_89"];
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for arch in arches {
        fs::create_dir_all(out_path.join(arch)).unwrap();
        for kernel in kernels {
            println!("cargo:rerun-if-changed=cpp/{}.cu", kernel);
            Command::new("nvcc")
                .arg(format!("-arch={}", arch))
                .arg("--cubin")
                .arg("-o")
                .arg(out_path.join(arch).join(format!("{}.cubin", kernel)))
                .arg(format!("cpp/{}.cu", kernel))
                .status()
                .unwrap();
        }
    }
}
