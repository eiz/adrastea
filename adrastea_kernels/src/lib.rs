#![allow(non_upper_case_globals)]

macro_rules! cuda_kernels {
    (@expand_arches [$($arch:ident),*] $kernel:ident) => {
        &[$(
            (stringify!($arch), include_bytes!(
                concat!(env!("OUT_DIR"), "/", stringify!($arch), "/", stringify!($kernel), ".cubin"))),
        )*]
    };
    (@expand_kernels $arches:tt [$($kernel:ident),*]) => {
        $(
            pub static $kernel: &[(& str, &[u8])] =
                cuda_kernels!(@expand_arches $arches $kernel);
        )*
    };
    ($arches:tt, $kernels:tt) => { cuda_kernels!(@expand_kernels $arches $kernels); };
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
        softmax_rows,
        square_fp32_16x16
    ]
}
