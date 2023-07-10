#![allow(non_upper_case_globals)]

macro_rules! simt_kernels {
    (@expand_arches [$($arch:ident : $arch_expr:expr),*] $kernel:ident) => {
        &[$(
            #[cfg(feature = $arch_expr)]
            (stringify!($arch), include_bytes!(
                concat!(env!("OUT_DIR"), "/", stringify!($arch), "/", stringify!($kernel), ".bin"))),
        )*]
    };
    (@expand_kernels $arches:tt [$($kernel:ident),*]) => {
        $(
            pub static $kernel: &[(&str, &[u8])] =
                simt_kernels!(@expand_arches $arches $kernel);
        )*
    };
    ($arches:tt, $kernels:tt) => { simt_kernels!(@expand_kernels $arches $kernels); };
}

simt_kernels! {
    [sm_80: "sm_80", sm_89: "sm_89", gfx1100: "gfx1100"],
    [
        convert,
        convolution,
        elementwise,
        embed,
        error_stats,
        matmul,
        microbench,
        normalize,
        quantize,
        rotary
    ]
}
