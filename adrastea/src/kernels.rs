use std::{collections::HashSet, os::raw::c_void};

use half::f16;
use parking_lot::Mutex;
use simt_hip::{HipModule, Kernel, KernelParam, LaunchParams};

use crate::{
    tensor::{Tensor, TensorLayout, TensorView, TensorViewMut},
    util::ceil_div,
};

fn encode_ptr(data: *mut c_void, n_dims: usize) -> u64 {
    let mut ptr = data as u64;
    assert!(n_dims <= 7);
    assert!(ptr & 0x7 == 0);
    ptr |= n_dims as u64;
    ptr
}

#[repr(C)]
pub struct TensorGpuDescriptor {
    pub ptr: u64,
    pub shape: [u32; 7],
    pub strides: [u32; 7],
}

impl KernelParam for TensorGpuDescriptor {}

impl TensorGpuDescriptor {
    pub fn as_gpu_ptr(&self) -> *mut c_void {
        (self.ptr & !0x7) as *mut c_void
    }
    pub fn n_dims(&self) -> usize {
        (self.ptr & 0x7) as usize
    }
}

impl<T: Copy> From<&TensorView<'_, T>> for TensorGpuDescriptor {
    fn from(tensor: &TensorView<T>) -> Self {
        let mut shape = [0; 7];
        let mut strides = [0; 7];
        for (i, (&dim, &stride)) in
            tensor.layout().dims.iter().rev().zip(tensor.layout().strides.iter().rev()).enumerate()
        {
            shape[i] = dim as u32;
            strides[i] = stride as u32;
        }
        Self {
            ptr: encode_ptr(tensor.as_gpu_ptr() as *const _ as *mut _, tensor.layout().dims.len()),
            shape,
            strides,
        }
    }
}

impl<T: Copy> From<&mut TensorViewMut<'_, T>> for TensorGpuDescriptor {
    fn from(tensor: &mut TensorViewMut<T>) -> Self {
        (&tensor.as_view()).into()
    }
}

#[repr(u32)]
pub enum Conv1dActivation {
    None = 0,
    Gelu = 1,
}

#[repr(u32)]
pub enum BinaryOp {
    Add = 1,
    SiluMul = 2,
}

#[repr(u32)]
pub enum MatmulLoadOp {
    Identity = 0,
    Scale = 1,
}

pub enum MatmulLoad {
    Identity,
    Scale(f32),
}

impl MatmulLoad {
    pub fn lower(&self) -> MatmulLoadOp {
        match self {
            MatmulLoad::Identity => MatmulLoadOp::Identity,
            MatmulLoad::Scale(_) => MatmulLoadOp::Scale,
        }
    }
}

#[repr(u32)]
pub enum MatmulStoreOp {
    Identity = 0,
    GeluBias = 1,
    BetaGeluBias = 2,
    BetaBias = 3,
    Scale = 4,
    Add = 5,
}

pub enum MatmulStore<'a> {
    Identity,
    GeluBias(&'a TensorView<'a, f16>),
    BetaGeluBias(f32, &'a TensorView<'a, f16>),
    BetaBias(f32, &'a TensorView<'a, f16>),
    Scale(f32),
    Add,
}

impl<'a> MatmulStore<'a> {
    pub fn lower(&self) -> MatmulStoreOp {
        match self {
            MatmulStore::Identity => MatmulStoreOp::Identity,
            MatmulStore::GeluBias(_) => MatmulStoreOp::GeluBias,
            MatmulStore::BetaGeluBias(_, _) => MatmulStoreOp::BetaGeluBias,
            MatmulStore::BetaBias(_, _) => MatmulStoreOp::BetaBias,
            MatmulStore::Scale(_) => MatmulStoreOp::Scale,
            MatmulStore::Add => MatmulStoreOp::Add,
        }
    }
}

#[repr(u32)]
#[derive(Eq, PartialEq, Copy, Clone)]
pub enum MatmulMask {
    None = 0,
    Causal = 1,
}

pub struct MatmulOptions<'a> {
    pub load: MatmulLoad,
    pub store: MatmulStore<'a>,
    pub mask: MatmulMask,
}

impl<'a> MatmulOptions<'a> {
    pub fn new() -> Self {
        Self { load: MatmulLoad::Identity, store: MatmulStore::Identity, mask: MatmulMask::None }
    }

    pub fn load(mut self, load: MatmulLoad) -> Self {
        self.load = load;
        self
    }

    pub fn store(mut self, store: MatmulStore<'a>) -> Self {
        self.store = store;
        self
    }

    pub fn mask(mut self, mask: MatmulMask) -> Self {
        self.mask = mask;
        self
    }
}

pub trait CommonKernels {
    // TODO: replace with conv2d, lol
    fn conv1d(
        &self, output: &mut TensorViewMut<f16>, input: &TensorView<f16>, weight: &TensorView<f16>,
        bias: &TensorView<f16>, stride: i32, padding: i32, activation: Conv1dActivation,
    ) -> anyhow::Result<()>;
    fn elementwise_binary_2d_f16_inplace(
        &self, inout_left: &mut TensorViewMut<f16>, right: &TensorView<f16>, op: BinaryOp,
    ) -> anyhow::Result<()>;
    fn layer_norm(
        &self, output: &mut TensorViewMut<f16>, input: &TensorView<f16>, weight: &TensorView<f16>,
        bias: &TensorView<f16>, eps: f32,
    ) -> anyhow::Result<()>;
    fn rms_norm(
        &self, output: &mut TensorViewMut<f16>, input: &TensorView<f16>, weight: &TensorView<f16>,
        eps: f32,
    ) -> anyhow::Result<()>;
    fn rotary_inplace(
        &self, inout: &mut TensorViewMut<f16>, n_heads: i32, pos_offset: i32, theta: f32,
    ) -> anyhow::Result<()>;
    fn matmul_f16_slow(
        &self, output: &mut TensorViewMut<f16>, left: &TensorView<f16>, right: &TensorView<f16>,
        options: MatmulOptions,
    ) -> anyhow::Result<()>;
    fn matmul_f16(
        &self, output: &mut TensorViewMut<f16>, left: &TensorView<f16>, right: &TensorView<f16>,
        options: MatmulOptions,
    ) -> anyhow::Result<()>;
    fn softmax_rows_inplace(
        &self, output: &mut TensorViewMut<f16>, temperature: f32,
    ) -> anyhow::Result<()>;
    fn embed(
        &self, output: &mut TensorViewMut<f16>, tokens: TensorView<i32>, embed: TensorView<f16>,
    ) -> anyhow::Result<()>;
}

pub struct MatmulTracer {
    kernels: GpuKernels,
    shape_set: Mutex<HashSet<(TensorLayout, TensorLayout, TensorLayout)>>,
}

impl MatmulTracer {
    pub fn new(kernels: GpuKernels) -> Self {
        Self { kernels, shape_set: Mutex::new(HashSet::new()) }
    }

    pub fn clear(&self) {
        self.shape_set.lock().clear();
    }

    pub fn shapes(&self) -> Vec<(TensorLayout, TensorLayout, TensorLayout)> {
        self.shape_set.lock().iter().cloned().collect()
    }
}

impl CommonKernels for MatmulTracer {
    fn conv1d(
        &self, output: &mut TensorViewMut<f16>, input: &TensorView<f16>, weight: &TensorView<f16>,
        bias: &TensorView<f16>, stride: i32, padding: i32, activation: Conv1dActivation,
    ) -> anyhow::Result<()> {
        self.kernels.conv1d(output, input, weight, bias, stride, padding, activation)
    }
    fn elementwise_binary_2d_f16_inplace(
        &self, inout_left: &mut TensorViewMut<f16>, right: &TensorView<f16>, op: BinaryOp,
    ) -> anyhow::Result<()> {
        self.kernels.elementwise_binary_2d_f16_inplace(inout_left, right, op)
    }
    fn layer_norm(
        &self, output: &mut TensorViewMut<f16>, input: &TensorView<f16>, weight: &TensorView<f16>,
        bias: &TensorView<f16>, eps: f32,
    ) -> anyhow::Result<()> {
        self.kernels.layer_norm(output, input, weight, bias, eps)
    }
    fn rms_norm(
        &self, output: &mut TensorViewMut<f16>, input: &TensorView<f16>, weight: &TensorView<f16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        self.kernels.rms_norm(output, input, weight, eps)
    }
    fn rotary_inplace(
        &self, inout: &mut TensorViewMut<f16>, n_heads: i32, pos_offset: i32, theta: f32,
    ) -> anyhow::Result<()> {
        self.kernels.rotary_inplace(inout, n_heads, pos_offset, theta)
    }
    fn matmul_f16_slow(
        &self, output: &mut TensorViewMut<f16>, left: &TensorView<f16>, right: &TensorView<f16>,
        options: MatmulOptions,
    ) -> anyhow::Result<()> {
        let left_layout = left.layout();
        let right_layout = right.layout();
        let output_layout = output.layout();
        self.shape_set.lock().insert((
            left_layout.clone(),
            right_layout.clone(),
            output_layout.clone(),
        ));
        self.kernels.matmul_f16_slow(output, left, right, options)
    }
    fn matmul_f16(
        &self, output: &mut TensorViewMut<f16>, left: &TensorView<f16>, right: &TensorView<f16>,
        options: MatmulOptions,
    ) -> anyhow::Result<()> {
        let left_layout = left.layout();
        let right_layout = right.layout();
        let output_layout = output.layout();
        self.shape_set.lock().insert((
            left_layout.clone(),
            right_layout.clone(),
            output_layout.clone(),
        ));
        self.kernels.matmul_f16(output, left, right, options)
    }
    fn softmax_rows_inplace(
        &self, output: &mut TensorViewMut<f16>, temperature: f32,
    ) -> anyhow::Result<()> {
        self.kernels.softmax_rows_inplace(output, temperature)
    }
    fn embed(
        &self, output: &mut TensorViewMut<f16>, tokens: TensorView<i32>, embed: TensorView<f16>,
    ) -> anyhow::Result<()> {
        self.kernels.embed(output, tokens, embed)
    }
}

pub struct GpuKernels {
    _modules: Vec<HipModule>,
    conv1d: Kernel<(
        TensorGpuDescriptor,
        TensorGpuDescriptor,
        TensorGpuDescriptor,
        TensorGpuDescriptor,
        i32,
        i32,
        i32,
    )>,
    layer_norm:
        Kernel<(*mut f16, *const f16, *const f16, *const f16, i32, i32, i32, i32, i32, i32, f32)>,
    rms_norm: Kernel<(*mut f16, *const f16, *const f16, i32, i32, i32, i32, i32, i32, f32)>,
    rotary: Kernel<(*mut f16, *const f16, i32, i32, i32, i32, i32, i32, i32, i32, f32)>,
    matmul_f16_slow: Kernel<(
        *mut f16,
        *const f16,
        *const f16,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const f16,
        f32,
        f32,
        u32,
        u32,
        u32,
    )>,
    matmul_f16: Kernel<(
        *mut f16,
        *const f16,
        *const f16,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const f16,
        f32,
        f32,
        u32,
        u32,
        u32,
    )>,
    elementwise_binary_2d_f16:
        Kernel<(*mut f16, *const f16, *const f16, i32, i32, i32, i32, i32, i32, i32, i32, u32)>,
    softmax_rows: Kernel<(*mut f16, *const f16, i32, i32, f32)>,
    embed: Kernel<(*mut f16, *const i32, i32, i32, *const f16)>,
}

impl GpuKernels {
    pub fn new(capability: i32) -> anyhow::Result<Self> {
        let module_conv1d = HipModule::find(capability, adrastea_kernels::conv1d)?;
        let module_layer_norm = HipModule::find(capability, adrastea_kernels::layer_norm)?;
        let module_rms_norm = HipModule::find(capability, adrastea_kernels::rms_norm)?;
        let module_rotary = HipModule::find(capability, adrastea_kernels::rotary)?;
        let module_elementwise = HipModule::find(capability, adrastea_kernels::elementwise)?;
        let module_matmul = HipModule::find(capability, adrastea_kernels::matmul)?;
        let module_softmax_rows = HipModule::find(capability, adrastea_kernels::softmax_rows)?;
        let module_embed = HipModule::find(capability, adrastea_kernels::embed)?;
        let kernels = GpuKernels {
            conv1d: Kernel::new(&module_conv1d, "conv1d")?,
            layer_norm: Kernel::new(&module_layer_norm, "layer_norm")?,
            rms_norm: Kernel::new(&module_rms_norm, "rms_norm")?,
            rotary: Kernel::new(&module_rotary, "rotary")?,
            matmul_f16_slow: Kernel::new(&module_matmul, "matmul_f16")?,
            matmul_f16: Kernel::new(&module_matmul, "matmul_f16_fast")?,
            elementwise_binary_2d_f16: Kernel::new(
                &module_elementwise,
                "elementwise_binary_2d_f16",
            )?,
            softmax_rows: Kernel::new(&module_softmax_rows, "softmax_rows")?,
            embed: Kernel::new(&module_embed, "embed")?,
            _modules: vec![
                module_conv1d,
                module_layer_norm,
                module_rms_norm,
                module_rotary,
                module_elementwise,
                module_matmul,
                module_softmax_rows,
                module_embed,
            ],
        };
        Ok(kernels)
    }
}

impl CommonKernels for GpuKernels {
    fn conv1d(
        &self, output: &mut TensorViewMut<f16>, input: &TensorView<f16>, weight: &TensorView<f16>,
        bias: &TensorView<f16>, stride: i32, padding: i32, activation: Conv1dActivation,
    ) -> anyhow::Result<()> {
        self.conv1d.launch(
            LaunchParams {
                blocks: (
                    ceil_div(input.size(-1) as u64, 16) as u32,
                    ceil_div(output.size(-2) as u64, 16) as u32,
                    1,
                ),
                threads: (16, 16, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                output.into(),
                input.into(),
                weight.into(),
                bias.into(),
                stride,
                padding,
                activation as i32,
            ),
        )?;
        Ok(())
    }

    fn elementwise_binary_2d_f16_inplace(
        &self, inout_left: &mut TensorViewMut<f16>, right: &TensorView<f16>, op: BinaryOp,
    ) -> anyhow::Result<()> {
        self.elementwise_binary_2d_f16.launch(
            LaunchParams {
                blocks: (
                    ceil_div(inout_left.size(-1) as u64, 16) as u32,
                    ceil_div(inout_left.size(-2) as u64, 16) as u32,
                    1,
                ),
                threads: (16, 16, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                inout_left.as_mut_gpu_ptr(),
                inout_left.as_gpu_ptr(),
                right.as_gpu_ptr(),
                inout_left.size(-1) as i32,
                inout_left.size(-2) as i32,
                inout_left.stride(-1) as i32,
                inout_left.stride(-2) as i32,
                inout_left.stride(-1) as i32,
                inout_left.stride(-2) as i32,
                right.stride(-1) as i32,
                right.stride(-2) as i32,
                op as u32,
            ),
        )?;
        Ok(())
    }

    fn layer_norm(
        &self, output: &mut TensorViewMut<f16>, input: &TensorView<f16>, weight: &TensorView<f16>,
        bias: &TensorView<f16>, eps: f32,
    ) -> anyhow::Result<()> {
        self.layer_norm.launch(
            LaunchParams {
                blocks: (input.size(-2) as u32, 1, 1),
                threads: (256, 1, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                output.as_mut_gpu_ptr(),
                input.as_gpu_ptr(),
                weight.as_gpu_ptr(),
                bias.as_gpu_ptr(),
                output.size(-1) as i32,
                output.size(-2) as i32,
                output.stride(-1) as i32,
                output.stride(-2) as i32,
                input.stride(-1) as i32,
                input.stride(-2) as i32,
                eps,
            ),
        )?;
        Ok(())
    }

    fn rms_norm(
        &self, output: &mut TensorViewMut<f16>, input: &TensorView<f16>, weight: &TensorView<f16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        assert_eq!(input.size(-1), output.size(-1));
        assert_eq!(input.size(-2), output.size(-2));
        self.rms_norm.launch(
            LaunchParams {
                blocks: (input.size(-2) as u32, 1, 1),
                threads: (256, 1, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                output.as_mut_gpu_ptr(),
                input.as_gpu_ptr(),
                weight.as_gpu_ptr(),
                output.size(-1) as i32,
                output.size(-2) as i32,
                output.stride(-1) as i32,
                output.stride(-2) as i32,
                input.stride(-1) as i32,
                input.stride(-2) as i32,
                eps,
            ),
        )?;
        Ok(())
    }

    fn rotary_inplace(
        &self, inout: &mut TensorViewMut<f16>, n_heads: i32, pos_offset: i32, theta: f32,
    ) -> anyhow::Result<()> {
        self.rotary.launch(
            LaunchParams {
                blocks: (
                    ceil_div(inout.size(-1) as u64, 16) as u32,
                    ceil_div(inout.size(-2) as u64, 16) as u32,
                    1,
                ),
                threads: (16, 16, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                inout.as_mut_gpu_ptr(),
                inout.as_gpu_ptr(),
                inout.size(-1) as i32,
                inout.size(-2) as i32,
                inout.stride(-1) as i32,
                inout.stride(-2) as i32,
                inout.stride(-1) as i32,
                inout.stride(-2) as i32,
                n_heads,
                pos_offset,
                theta,
            ),
        )?;
        Ok(())
    }

    fn matmul_f16_slow(
        &self, output: &mut TensorViewMut<f16>, left: &TensorView<f16>, right: &TensorView<f16>,
        options: MatmulOptions,
    ) -> anyhow::Result<()> {
        assert_eq!(left.size(-1), right.size(-2)); // K
        assert_eq!(output.size(-2), left.size(-2)); // M
        assert_eq!(output.size(-1), right.size(-1)); // N
        let bias = match &options.store {
            MatmulStore::Identity => None,
            MatmulStore::GeluBias(bias) => Some(bias),
            MatmulStore::BetaGeluBias(_, bias) => Some(bias),
            MatmulStore::BetaBias(_, bias) => Some(bias),
            MatmulStore::Scale(_) => None,
            MatmulStore::Add => None,
        };
        let mut scale = match &options.load {
            MatmulLoad::Identity => 1.0,
            MatmulLoad::Scale(scale) => *scale,
        };
        if let MatmulStore::Scale(store_scale) = &options.store {
            scale *= store_scale;
        };
        let beta = match &options.store {
            MatmulStore::BetaGeluBias(beta, _) => *beta,
            MatmulStore::BetaBias(beta, _) => *beta,
            _ => 0.0,
        };
        let bias_ptr = bias.map(|b| b.as_gpu_ptr()).unwrap_or(std::ptr::null());
        self.matmul_f16_slow.launch(
            LaunchParams {
                blocks: (
                    ceil_div(output.size(-1) as u64, 16) as u32,
                    ceil_div(output.size(-2) as u64, 16) as u32,
                    if output.layout().dims.len() > 2 { output.size(-3) as u32 } else { 1 },
                ),
                threads: (16, 16, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                output.as_mut_gpu_ptr(),
                left.as_gpu_ptr(),
                right.as_gpu_ptr(),
                if output.layout().dims.len() > 2 { output.size(-3) as i32 } else { 1 },
                left.size(-2) as i32,
                left.size(-1) as i32,
                right.size(-1) as i32,
                output.stride(-1) as i32,
                output.stride(-2) as i32,
                if output.layout().dims.len() > 2 { output.stride(-3) } else { 0 } as i32,
                left.stride(-1) as i32,
                left.stride(-2) as i32,
                if left.layout().dims.len() > 2 { left.stride(-3) } else { 0 } as i32,
                right.stride(-1) as i32,
                right.stride(-2) as i32,
                if right.layout().dims.len() > 2 { right.stride(-3) } else { 0 } as i32,
                bias_ptr,
                beta,
                scale,
                options.store.lower() as u32,
                options.load.lower() as u32,
                options.mask as u32,
            ),
        )?;
        Ok(())
    }

    fn matmul_f16(
        &self, output: &mut TensorViewMut<f16>, left: &TensorView<f16>, right: &TensorView<f16>,
        options: MatmulOptions,
    ) -> anyhow::Result<()> {
        assert_eq!(left.size(-1), right.size(-2)); // K
        assert_eq!(output.size(-2), left.size(-2)); // M
        assert_eq!(output.size(-1), right.size(-1)); // N
        let bias = match &options.store {
            MatmulStore::Identity => None,
            MatmulStore::GeluBias(bias) => Some(bias),
            MatmulStore::BetaGeluBias(_, bias) => Some(bias),
            MatmulStore::BetaBias(_, bias) => Some(bias),
            MatmulStore::Scale(_) => None,
            MatmulStore::Add => None,
        };
        let mut scale = match &options.load {
            MatmulLoad::Identity => 1.0,
            MatmulLoad::Scale(scale) => *scale,
        };
        if let MatmulStore::Scale(store_scale) = &options.store {
            scale *= store_scale;
        };
        let beta = match &options.store {
            MatmulStore::BetaGeluBias(beta, _) => *beta,
            MatmulStore::BetaBias(beta, _) => *beta,
            _ => 0.0,
        };
        const K_TILE: u32 = 32;
        const N_TILE: u32 = 32;
        const M_TILE: u32 = 16;
        let bias_ptr = bias.map(|b| b.as_gpu_ptr()).unwrap_or(std::ptr::null());
        self.matmul_f16.launch(
            LaunchParams {
                blocks: (
                    ceil_div(output.size(-1) as u64, N_TILE as u64) as u32,
                    ceil_div(output.size(-2) as u64, M_TILE as u64) as u32,
                    if output.layout().dims.len() > 2 { output.size(-3) as u32 } else { 1 },
                ),
                threads: (32, 4, 1),
                shared_mem: M_TILE * (K_TILE + 2) * 2 + K_TILE * (N_TILE + 2) * 2,
                stream: None,
            },
            (
                output.as_mut_gpu_ptr(),
                left.as_gpu_ptr(),
                right.as_gpu_ptr(),
                if output.layout().dims.len() > 2 { output.size(-3) as i32 } else { 1 },
                left.size(-2) as i32,
                left.size(-1) as i32,
                right.size(-1) as i32,
                output.stride(-1) as i32,
                output.stride(-2) as i32,
                if output.layout().dims.len() > 2 { output.stride(-3) } else { 0 } as i32,
                left.stride(-1) as i32,
                left.stride(-2) as i32,
                if left.layout().dims.len() > 2 { left.stride(-3) } else { 0 } as i32,
                right.stride(-1) as i32,
                right.stride(-2) as i32,
                if right.layout().dims.len() > 2 { right.stride(-3) } else { 0 } as i32,
                bias_ptr,
                beta,
                scale,
                options.store.lower() as u32,
                options.load.lower() as u32,
                options.mask as u32,
            ),
        )?;
        Ok(())
    }

    fn softmax_rows_inplace(
        &self, output: &mut TensorViewMut<f16>, temperature: f32,
    ) -> anyhow::Result<()> {
        self.softmax_rows.launch(
            LaunchParams {
                blocks: (
                    output.size(-2) as u32,
                    if output.layout().dims.len() > 2 { output.size(-3) as u32 } else { 1 },
                    1,
                ),
                threads: (256, 1, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                output.as_mut_gpu_ptr(),
                output.as_gpu_ptr(),
                output.size(-2) as i32,
                output.size(-1) as i32,
                temperature,
            ),
        )?;
        Ok(())
    }

    fn embed(
        &self, output: &mut TensorViewMut<f16>, tokens: TensorView<i32>, embed: TensorView<f16>,
    ) -> anyhow::Result<()> {
        assert_eq!(tokens.size(-1), output.size(-2));
        assert_eq!(output.size(-1), embed.size(-1));
        self.embed.launch(
            LaunchParams {
                blocks: (ceil_div(output.size(-2) as u64, 1024) as u32, 1, 1),
                threads: (1024, 1, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                output.as_mut_gpu_ptr(),
                tokens.as_gpu_ptr(),
                output.size(-1) as i32,
                output.size(-2) as i32,
                embed.as_gpu_ptr(),
            ),
        )?;
        Ok(())
    }
}
