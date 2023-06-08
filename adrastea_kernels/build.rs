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
            .arg("-O2")
            .arg("-Wall")
            .arg("-Werror")
            .arg("-o")
            .arg(out_path.join(arch).join(format!("{}.bin", kernel)))
            .arg(format!("cpp/{}.cu", kernel))
            .status()
            .unwrap();
        if !status.success() {
            panic!("failed to compile {}", kernel);
        }
        let status = Command::new("hipcc")
            .arg("--genco")
            .arg(format!("--offload-arch={}", arch))
            .arg("-O2")
            .arg("-S")
            .arg("-o")
            .arg(out_path.join(arch).join(format!("{}.S", kernel)))
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
        "conv1d",
        "elementwise",
        "embed",
        "embed_uint8",
        "layer_norm",
        "matmul",
        "matmul_nt_fp16u8",
        "microbench",
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
    #[allow(unused_mut)]
    let mut cuda_arches: Vec<&str> = vec![];
    #[allow(unused_mut)]
    let mut hip_arches: Vec<&str> = vec![];
    #[cfg(feature = "sm_80")]
    cuda_arches.push("sm_80");
    #[cfg(feature = "sm_89")]
    cuda_arches.push("sm_89");
    #[cfg(feature = "gfx1100")]
    hip_arches.push("gfx1100");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for arch in cuda_arches {
        build_arch_cuda(&out_path, arch, kernels);
        build_arch_cuda(&out_path, arch, kernels_cuda);
    }
    for arch in hip_arches {
        build_arch_hip(&out_path, arch, kernels);
    }
}
