#![allow(non_upper_case_globals)]

macro_rules! cuda_kernels {
    (@arch $arch:ident [$($kernel:ident),*]) => {
        $(
            pub static $kernel: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", stringify!($arch), "/", stringify!($kernel), ".cubin"));
        )*
    };
    ([$($arch:ident),*], $kernels:tt) => {
        $(
            pub mod $arch {
                cuda_kernels!{@arch $arch $kernels}
            }
        )*
    };
}

cuda_kernels! {
    [sm_80, sm_89],
    [
        embed,
        embed_uint8,
        matmul_nt_fp16u8,
        matmul_nt_wmma_16x128x256_fp16u8,
        matmul_nt_wmma_16x128x256,
        matmul_nt_wmma_128x64x64,
        matmul_nt,
        matmul_qk,
        matmul_qkv,
        rms_norm,
        rotary,
        silu,
        softmax_rows
    ]
}
