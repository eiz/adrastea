use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn build_arch_cuda(out_path: &Path, arch: &str, kernels: &[&str]) {
    fs::create_dir_all(out_path.join(arch)).unwrap();
    for kernel in kernels {
        println!("cargo:rerun-if-changed=cpp/{}.cu", kernel);
        let status = Command::new("nvcc")
            .arg(format!("-arch={}", arch))
            .arg("--cubin")
            .arg("-o")
            .arg(out_path.join(arch).join(format!("{}.bin", kernel)))
            .arg(format!("cpp/{}.cu", kernel))
            .status()
            .unwrap();
        if !status.success() {
            panic!("failed to compile {}", kernel);
        }
    }
}

fn build_arch_hip(out_path: &Path, arch: &str, kernels: &[&str]) {
    fs::create_dir_all(out_path.join(arch)).unwrap();
    for kernel in kernels {
        println!("cargo:rerun-if-changed=cpp/{}.cu", kernel);
        let status = Command::new("hipcc")
            .arg("--genco")
            .arg(format!("--offload-arch={}", arch))
            .arg("-o")
            .arg(out_path.join(arch).join(format!("{}.bin", kernel)))
            .arg(format!("cpp/{}.cu", kernel))
            .status()
            .unwrap();
        if !status.success() {
            panic!("failed to compile {}", kernel);
        }
    }
}

fn main() {
    let kernels = &[
        "embed",
        "embed_uint8",
        "matmul_nt_fp16u8",
        "matmul_nt",
        "matmul_qk",
        "matmul_qkv",
        "rms_norm",
        "rotary",
        "silu",
        "softmax_rows",
        "square_fp32_16x16",
    ];
    let kernels_cuda = &[
        "matmul_nt_wmma_16x128x256_fp16u8",
        "matmul_nt_wmma_16x128x256",
        "matmul_nt_wmma_128x64x64",
    ];
    let cuda_arches = &["sm_80", "sm_89"];
    let hip_arches = &["gfx1100"];
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for arch in cuda_arches {
        build_arch_cuda(&out_path, arch, kernels);
        build_arch_cuda(&out_path, arch, kernels_cuda);
    }
    for arch in hip_arches {
        build_arch_hip(&out_path, arch, kernels);
    }
}
