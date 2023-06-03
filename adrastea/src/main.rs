/*
 * This file is part of Adrastea.
 *
 * Adrastea is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Affero General Public License as published by the Free Software
 * Foundation, version 3.
 *
 * Adrastea is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along
 * with Adrastea. If not, see <https://www.gnu.org/licenses/>.
 */

use core::{
    cell::RefCell,
    ffi::{c_void, CStr},
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};
use std::{collections::HashMap, fs::File, path::Path, time::Instant};

use alloc::sync::Arc;
use anyhow::bail;
use ash::{vk, Entry};
use half::f16;
use rustfft::num_complex::Complex32;
use serde::{Deserialize, Serialize};
use simt_hip::{
    HipBuffer, HipDevice, HipModule, HipPhysicalDevice, HipStream, Kernel, LaunchParams,
};
use smallvec::SmallVec;
use tiktoken_rs::CoreBPE;

use crate::pickle::{ModelState, PickledModel};

extern crate alloc;

pub mod mel;
pub mod pickle;
pub mod stft;

const THEM_SHADERS: &[u8] = include_bytes!("../../shaders/square.comp.spv");

fn ceil_div(a: vk::DeviceSize, b: vk::DeviceSize) -> vk::DeviceSize {
    (a + b - 1) / b
}

unsafe fn find_compute_queue_family(instance: &ash::Instance, phys_dev: vk::PhysicalDevice) -> u32 {
    let props = instance.get_physical_device_queue_family_properties(phys_dev);

    for (i, fam) in props.iter().enumerate() {
        if fam.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && fam.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && fam.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        {
            return i as u32;
        }
    }

    panic!("oh noes couldn't find a compute queue");
}

unsafe fn find_memory_type(
    reqs: &vk::MemoryRequirements, mem_props: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> u32 {
    for i in 0..mem_props.memory_type_count {
        let t = &mem_props.memory_types[i as usize];
        if (reqs.memory_type_bits & (1 << i)) != 0 && (t.property_flags & flags) == flags {
            return i;
        }
    }

    panic!("unable to find a suitable memory type")
}

#[repr(C)]
struct SquareArgs {
    in_buf: u64,
    out_buf: u64,
    height: u32,
    width: u32,
}

const ROWS: vk::DeviceSize = 8;
const COLS: vk::DeviceSize = 8;

unsafe fn vulkan_square() -> anyhow::Result<()> {
    let entry = Entry::load()?;
    let app_info =
        vk::ApplicationInfo { api_version: vk::make_api_version(0, 1, 3, 0), ..Default::default() };
    let instance = entry.create_instance(
        &vk::InstanceCreateInfo::builder().application_info(&app_info),
        //.enabled_layer_names(&[b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8]),
        None,
    )?;
    dbg!(instance.handle());
    let phys_devs = instance.enumerate_physical_devices()?;
    dbg!(&phys_devs);
    let queue_family_index = find_compute_queue_family(&instance, phys_devs[0]);
    let queue_infos = [vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0])
        .build()];
    let mut create_info = vk::DeviceCreateInfo {
        queue_create_info_count: 1,
        p_queue_create_infos: queue_infos.as_ptr(),
        ..Default::default()
    };
    let mut phys_features_13: vk::PhysicalDeviceVulkan13Features = Default::default();
    let mut phys_features_12: vk::PhysicalDeviceVulkan12Features = Default::default();
    let mut phys_features = vk::PhysicalDeviceFeatures2::builder()
        .push_next(&mut phys_features_13)
        .push_next(&mut phys_features_12)
        .build();
    instance.get_physical_device_features2(phys_devs[0], &mut phys_features);
    if phys_features_13.synchronization2 == 0 {
        panic!("synchronization2 missing");
    }
    if phys_features_12.buffer_device_address == 0 {
        panic!("buffer_device_address missing");
    }
    let mut enabled_features_13 =
        vk::PhysicalDeviceVulkan13Features::builder().synchronization2(true).build();
    let mut enabled_features_12 =
        vk::PhysicalDeviceVulkan12Features::builder().buffer_device_address(true).build();
    let enabled_features = vk::PhysicalDeviceFeatures2::builder()
        .push_next(&mut enabled_features_13)
        .push_next(&mut enabled_features_12)
        .build();
    create_info.p_next = &enabled_features as *const _ as *mut _;
    println!("features_12: {:?}", phys_features_12);
    println!("features_13: {:?}", phys_features_13);
    let dev = instance.create_device(phys_devs[0], &create_info, None)?;
    dbg!(dev.handle());
    let fence = dev.create_fence(&Default::default(), None)?;
    let queue = dev.get_device_queue(queue_family_index, 0);
    dbg!(&queue);
    let command_pool = dev.create_command_pool(
        &vk::CommandPoolCreateInfo {
            queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        },
        None,
    )?;
    dbg!(&command_pool);
    let bufs = dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    })?;
    dbg!(&bufs);
    let shader = dev.create_shader_module(
        &vk::ShaderModuleCreateInfo {
            code_size: THEM_SHADERS.len(),
            p_code: THEM_SHADERS.as_ptr() as *const _,
            ..Default::default()
        },
        None,
    )?;
    dbg!(&shader);
    let p_create_info = vk::PipelineLayoutCreateInfo::builder()
        .push_constant_ranges(&[vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(24)
            .build()])
        .build();
    let p_layout = dev.create_pipeline_layout(&p_create_info, None)?;
    dbg!(&p_layout);
    let p_cache =
        dev.create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder().build(), None)?;
    let pipelines = dev
        .create_compute_pipelines(
            p_cache,
            &[vk::ComputePipelineCreateInfo::builder()
                .layout(p_layout)
                .stage(
                    vk::PipelineShaderStageCreateInfo::builder()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader)
                        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
                        .build(),
                )
                .build()],
            None,
        )
        .expect("derp");
    dbg!(&pipelines);
    let buf = dev.create_buffer(
        &vk::BufferCreateInfo::builder()
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .size(ROWS * COLS * 4),
        None,
    )?;
    dbg!(&buf);
    let stage_buf = dev.create_buffer(
        &vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .size(ROWS * COLS * 4),
        None,
    )?;
    dbg!(&stage_buf);
    let mem_props = instance.get_physical_device_memory_properties(phys_devs[0]);
    dbg!(&mem_props);
    let buf_reqs = dev.get_buffer_memory_requirements(buf);
    let stage_reqs = dev.get_buffer_memory_requirements(stage_buf);
    dbg!(&buf_reqs);
    dbg!(&stage_reqs);
    let default_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let stage_flags =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let buf_mem = dev.allocate_memory(
        &vk::MemoryAllocateInfo::builder()
            .memory_type_index(find_memory_type(&buf_reqs, &mem_props, default_flags))
            .allocation_size(buf_reqs.size)
            .push_next(
                &mut vk::MemoryAllocateFlagsInfo::builder()
                    .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS),
            ),
        None,
    )?;
    let stage_mem = dev.allocate_memory(
        &vk::MemoryAllocateInfo::builder()
            .memory_type_index(find_memory_type(&stage_reqs, &mem_props, stage_flags))
            .allocation_size(stage_reqs.size),
        None,
    )?;
    dbg!(&buf_mem);
    dbg!(&stage_mem);
    dev.bind_buffer_memory(buf, buf_mem, 0)?;
    dev.bind_buffer_memory(stage_buf, stage_mem, 0)?;
    let ptr = dev.map_memory(stage_mem, 0, stage_reqs.size, Default::default())?;
    {
        let in_float: &mut [f32] =
            std::slice::from_raw_parts_mut(ptr as *mut _, (stage_reqs.size / 4) as usize);
        for y in 0..ROWS {
            for x in 0..COLS {
                in_float[(y * COLS + x) as usize] = (y + x) as f32;
            }
        }
    }

    // oh my god fucking do the thing already
    let cmd = bufs[0];
    dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo { ..Default::default() })?;
    dev.cmd_copy_buffer(
        cmd,
        stage_buf,
        buf,
        &[vk::BufferCopy::builder().size(ROWS * COLS * 4).build()],
    );
    dev.cmd_pipeline_barrier2(
        cmd,
        &vk::DependencyInfo::builder().buffer_memory_barriers(&[
            vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .buffer(buf)
                .size(vk::WHOLE_SIZE)
                .build(),
        ]),
    );
    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipelines[0]);
    let args = SquareArgs {
        in_buf: dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::builder().buffer(buf)),
        out_buf: dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::builder().buffer(buf)),
        height: ROWS as u32,
        width: COLS as u32,
    };
    dev.cmd_push_constants(
        cmd,
        p_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        // lord forgive me lmao
        std::slice::from_raw_parts(
            &args as *const _ as *const _,
            std::mem::size_of::<SquareArgs>(),
        ),
    );
    dev.cmd_dispatch(cmd, ceil_div(COLS, 16) as u32, ceil_div(ROWS, 16) as u32, 1);
    dev.cmd_pipeline_barrier2(
        cmd,
        &vk::DependencyInfo::builder().buffer_memory_barriers(&[
            vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .buffer(buf)
                .size(vk::WHOLE_SIZE)
                .build(),
        ]),
    );
    dev.cmd_copy_buffer(
        cmd,
        buf,
        stage_buf,
        &[vk::BufferCopy::builder().size(ROWS * COLS * 4).build()],
    );
    dev.end_command_buffer(cmd)?;
    dev.queue_submit2(
        queue,
        &[vk::SubmitInfo2::builder()
            .command_buffer_infos(&[vk::CommandBufferSubmitInfo::builder()
                .command_buffer(cmd)
                .build()])
            .build()],
        fence,
    )
    .expect("derp");
    dev.wait_for_fences(&[fence], true, !0).expect("derp");
    {
        let out_float: &[f32] =
            std::slice::from_raw_parts(ptr as *mut _, (stage_reqs.size / 4) as usize);
        for y in 0..ROWS {
            for x in 0..COLS {
                print!("{:4} ", out_float[(y * COLS + x) as usize]);
            }
            println!("");
        }
    }
    Ok(())
    // the aristocrats.

    // unsafe fn dadada(out: *mut f32, lhs: *const half, rhs: *const half, m: u32, n: u32, k: u32, ) {
    // dadada(1024, 256, 0, stream)(out, lhs, rhs, m, n, k);
    // foo⟪1024, 256, 0, stream⟫(0);
}

#[inline(always)]
unsafe fn cuda_call<F: FnOnce() -> simt_cuda_sys::CUresult>(
    cb: F,
) -> Result<(), simt_cuda_sys::CUresult> {
    let res = cb();
    if res == simt_cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(res)
    }
}

#[inline(always)]
unsafe fn cuda_result_call<T, F: FnOnce(*mut T) -> simt_cuda_sys::CUresult>(
    cb: F,
) -> Result<T, simt_cuda_sys::CUresult> {
    let mut out = std::mem::MaybeUninit::uninit();
    let res = cb(out.as_mut_ptr());
    if res == simt_cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(out.assume_init())
    } else {
        Err(res)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TritonKernelMetadata {
    num_warps: u32,
    num_stages: u32,
    constants: HashMap<String, u32>,
    debug: bool,
    shared: u32,
    name: String,
}

struct CudaContext {
    cuda: Arc<simt_cuda_sys::cuda>,
    device: simt_cuda_sys::CUdevice,
    context: simt_cuda_sys::CUcontext,
}

impl CudaContext {
    pub unsafe fn new(cuda: Arc<simt_cuda_sys::cuda>, device_index: i32) -> anyhow::Result<Self> {
        let device = cuda_result_call(|x| cuda.cuDeviceGet(x, device_index))?;
        let context = cuda_result_call(|x| cuda.cuCtxCreate_v2(x, 0, device))?;
        cuda_result_call(|x| cuda.cuCtxPopCurrent_v2(x)).expect("cuCtxPopCurrent_v2 failed");
        Ok(Self { cuda, device, context })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            cuda_call(|| self.cuda.cuCtxDestroy_v2(self.context)).expect("cuCtxDestroy_v2 failed");
        }
    }
}

// exercise for the reader: make get() not have to return a clone
thread_local! {
    static THREAD_CUDA_CONTEXT: RefCell<Option<Arc<CudaContext>>> = RefCell::new(None);
}

struct ScopedCudaContext {
    old_value: Option<Arc<CudaContext>>,
}

impl ScopedCudaContext {
    unsafe fn new(value: Arc<CudaContext>) -> Result<Self, simt_cuda_sys::CUresult> {
        let old_value = THREAD_CUDA_CONTEXT.with(|v| {
            let mut v = v.borrow_mut();
            let old_value = v.clone();
            cuda_call(|| value.cuda.cuCtxPushCurrent_v2(value.context))?;
            *v = Some(value);
            Ok(old_value)
        })?;
        Ok(ScopedCudaContext { old_value })
    }

    fn get() -> Result<Arc<CudaContext>, simt_cuda_sys::CUresult> {
        THREAD_CUDA_CONTEXT
            .with(|v| v.borrow().clone())
            .ok_or(simt_cuda_sys::CUresult::CUDA_ERROR_INVALID_CONTEXT)
    }

    pub fn capability() -> Result<i32, simt_cuda_sys::CUresult> {
        unsafe {
            let ctx = ScopedCudaContext::get()?;
            let major = cuda_result_call(|x| {
                ctx.cuda.cuDeviceGetAttribute(
                    x,
                    simt_cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    ctx.device,
                )
            })?;
            let minor = cuda_result_call(|x| {
                ctx.cuda.cuDeviceGetAttribute(
                    x,
                    simt_cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    ctx.device,
                )
            })?;
            Ok(major * 10 + minor)
        }
    }
}

impl Drop for ScopedCudaContext {
    fn drop(&mut self) {
        THREAD_CUDA_CONTEXT.with(|v| {
            let mut v = v.borrow_mut();
            unsafe {
                cuda_result_call(|x| v.as_ref().unwrap().cuda.cuCtxPopCurrent_v2(x))
                    .expect("cuCtxPopCurrent_v2 failed");
            }
            *v = self.old_value.clone()
        });
    }
}

struct CudaBuffer {
    ptr: simt_cuda_sys::CUdeviceptr,
    size: usize,
}

impl CudaBuffer {
    pub unsafe fn new(size: usize) -> anyhow::Result<Self> {
        let ctx = ScopedCudaContext::get()?;
        let ptr = cuda_result_call(|x| ctx.cuda.cuMemAlloc_v2(x, size))?;
        Ok(Self { ptr, size })
    }

    pub unsafe fn copy_from(
        &mut self, src: *const std::ffi::c_void, size: usize,
    ) -> anyhow::Result<()> {
        let ctx = ScopedCudaContext::get()?;
        cuda_call(|| ctx.cuda.cuMemcpyHtoD_v2(self.ptr, src, size))?;
        Ok(())
    }

    pub unsafe fn copy_to(&self, dst: *mut std::ffi::c_void, size: usize) -> anyhow::Result<()> {
        let ctx = ScopedCudaContext::get()?;
        cuda_call(|| ctx.cuda.cuMemcpyDtoH_v2(dst, self.ptr, size))?;
        Ok(())
    }

    pub fn copy_from_slice<T: Copy>(&mut self, src: &[T]) -> anyhow::Result<()> {
        assert_eq!(src.len() * std::mem::size_of::<T>(), self.size);
        unsafe { self.copy_from(src.as_ptr() as *const std::ffi::c_void, self.size) }
    }

    pub fn copy_to_slice<T: Copy>(&self, dst: &mut [T]) -> anyhow::Result<()> {
        assert_eq!(dst.len() * std::mem::size_of::<T>(), self.size);
        unsafe { self.copy_to(dst.as_mut_ptr() as *mut std::ffi::c_void, self.size) }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            let ctx = ScopedCudaContext::get()
                .expect("invariant: CudaBuffer must be dropped in a context scope");
            cuda_call(|| ctx.cuda.cuMemFree_v2(self.ptr)).expect("cuMemFree_v2 failed");
        }
    }
}

pub struct CudaModule {
    inner: simt_cuda_sys::CUmodule,
}

impl CudaModule {
    pub unsafe fn new(data: &[u8]) -> anyhow::Result<Self> {
        let ctx = ScopedCudaContext::get()?;
        let inner = cuda_result_call(|x| ctx.cuda.cuModuleLoadData(x, data.as_ptr() as *const _))?;
        Ok(Self { inner })
    }

    pub unsafe fn find(capability: i32, kernels: &[(&str, &[u8])]) -> anyhow::Result<Self> {
        let ctx = ScopedCudaContext::get()?;
        let mut compatible_kernels = vec![];
        for (arch, bin) in kernels {
            if !arch.starts_with("sm_") {
                continue;
            }
            let arch = arch[3..].parse::<i32>()?;
            if arch <= capability {
                compatible_kernels.push((arch, bin));
            }
        }
        compatible_kernels.sort_by_key(|(arch, _)| *arch);
        let (_, bin) = compatible_kernels
            .iter()
            .rev()
            .filter(|(arch, _)| *arch <= capability)
            .last()
            .ok_or_else(|| anyhow::anyhow!("no compatible kernel found"))?;
        let inner = cuda_result_call(|x| ctx.cuda.cuModuleLoadData(x, bin.as_ptr() as *const _))?;
        Ok(Self { inner })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        unsafe {
            let ctx = ScopedCudaContext::get()
                .expect("invariant: CudaModule must be dropped in a context scope");
            cuda_call(|| ctx.cuda.cuModuleUnload(self.inner)).expect("cuModuleUnload failed");
        }
    }
}

unsafe fn cuda_square() -> anyhow::Result<()> {
    #[cfg(target_os = "linux")]
    const LIB: &str = "libcuda.so";
    #[cfg(windows)]
    const LIB: &str = "nvcuda.dll";
    let cuda = Arc::new(simt_cuda_sys::cuda::new(LIB)?);
    cuda_call(|| cuda.cuInit(0))?;
    let device_count = cuda_result_call(|x| cuda.cuDeviceGetCount(x))?;
    println!("{} device(s)", device_count);
    for i in 0..device_count {
        let mut name = [0u8; 256];
        cuda_call(|| cuda.cuDeviceGetName(name.as_mut_ptr() as *mut _, 256, i))?;
        let c_name = CStr::from_ptr(name.as_ptr() as *const _);
        println!("Device {}: {}", i, c_name.to_str()?);
    }
    if device_count == 0 {
        bail!("can't continue, no devices");
    }
    let context = Arc::new(CudaContext::new(cuda.clone(), 0)?);
    let _scoped_ctx = ScopedCudaContext::new(context.clone());
    let capability = ScopedCudaContext::capability()?;
    let module = CudaModule::find(capability, adrastea_kernels::square_fp32_16x16)?;
    let kernel = cuda_result_call(|x| {
        cuda.cuModuleGetFunction(x, module.inner, b"square_fp32_16x16\0".as_ptr() as *const i8)
    })?;
    dbg!(kernel);
    let stream = cuda_result_call(|x| cuda.cuStreamCreate(x, 0))?;
    dbg!(stream);
    let mut stage_buf = vec![0.0f32; (COLS * ROWS) as usize];
    let buf_sz = (COLS * ROWS * std::mem::size_of::<f32>() as u64) as usize;
    for y in 0..ROWS {
        for x in 0..COLS {
            stage_buf[(y * COLS + x) as usize] = (y + x) as f32;
        }
    }
    let mut buf = CudaBuffer::new(buf_sz)?;
    buf.copy_from_slice(&stage_buf)?;
    let grid_x = ceil_div(COLS, 16);
    let grid_y = ceil_div(ROWS, 16);
    let width = COLS as u32;
    let height = ROWS as u32;
    cuda_call(|| {
        cuda.cuLaunchKernel(
            kernel,
            grid_x as u32,
            grid_y as u32,
            1,
            16,
            16,
            1,
            0,
            stream,
            &[
                &buf.ptr as *const _ as *mut c_void,
                &buf.ptr as *const _ as *mut c_void,
                &width as *const _ as *mut c_void,
                &height as *const _ as *mut c_void,
            ] as *const _ as *mut _,
            std::ptr::null_mut(),
        )
    })?;
    cuda_call(|| cuda.cuStreamSynchronize(stream))?;
    buf.copy_to_slice(&mut stage_buf)?;
    for y in 0..ROWS {
        for x in 0..COLS {
            print!("{:4} ", stage_buf[(y * COLS + x) as usize]);
        }
        println!("");
    }
    Ok(())
}

struct TestKernels {
    square_fp32_16x16: Kernel<(*mut f32, *mut f32, u32, u32)>,
}

fn hip_square() -> anyhow::Result<()> {
    let device_count = HipPhysicalDevice::count()?;
    println!("{} device(s)", device_count);
    for i in 0..device_count {
        println!("Device {}: {}", i, HipPhysicalDevice::get(i)?.name()?);
    }
    if device_count == 0 {
        bail!("can't continue, no devices");
    }
    let phys = HipPhysicalDevice::get(0)?;
    let device = Arc::new(HipDevice::new(phys)?);
    let _scope = device.lock()?;
    let capability = phys.capability()?;
    println!("capability is {}", capability);
    let module_square_fp32_16x16 =
        HipModule::find(capability, adrastea_kernels::square_fp32_16x16)?;
    let kernels = TestKernels {
        square_fp32_16x16: Kernel::new(&module_square_fp32_16x16, "square_fp32_16x16")?,
    };
    let stream = HipStream::new()?;
    let mut stage_buf = vec![0.0f32; (COLS * ROWS) as usize];
    let buf_sz = (COLS * ROWS * std::mem::size_of::<f32>() as u64) as usize;
    for y in 0..ROWS {
        for x in 0..COLS {
            stage_buf[(y * COLS + x) as usize] = (y + x) as f32;
        }
    }
    let mut buf = HipBuffer::new(buf_sz)?;
    buf.copy_from_slice(&stage_buf)?;
    let grid_x = ceil_div(COLS, 16);
    let grid_y = ceil_div(ROWS, 16);
    kernels.square_fp32_16x16.launch(
        LaunchParams {
            blocks: (grid_x as u32, grid_y as u32, 1),
            threads: (16, 16, 1),
            shared_mem: 0,
            stream: Some(&stream),
        },
        (buf.ptr as *mut f32, buf.ptr as *mut f32, COLS as u32, ROWS as u32),
    )?;
    stream.sync()?;
    buf.copy_to_slice(&mut stage_buf)?;
    for y in 0..ROWS {
        for x in 0..COLS {
            print!("{:4} ", stage_buf[(y * COLS + x) as usize]);
        }
        println!("");
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct TensorLayout {
    pub dims: SmallVec<[usize; 7]>,
    pub strides: SmallVec<[usize; 7]>,
}

impl TensorLayout {
    pub fn new(dims: &[usize], strides: &[usize]) -> Self {
        assert_eq!(dims.len(), strides.len());
        assert_ne!(dims.len(), 0);
        assert_ne!(strides.len(), 0);
        assert!(dims.iter().all(|&x| x != 0));
        assert!(strides.iter().all(|&x| x != 0));
        Self { dims: dims.into(), strides: strides.into() }
    }

    pub fn row_major(dims: &[usize]) -> Self {
        let mut strides = SmallVec::<[usize; 7]>::new();
        let mut stride = 1;
        for &dim in dims.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();
        Self::new(dims, &strides)
    }

    pub fn largest_address(&self) -> usize {
        let mut addr = 0;
        for (&dim, &stride) in self.dims.iter().zip(self.strides.iter()) {
            addr += (dim - 1) * stride;
        }
        addr
    }

    pub fn permute(&self, dim_order: &[usize]) -> Self {
        assert_eq!(dim_order.len(), self.dims.len());
        let mut dims = SmallVec::<[usize; 7]>::new();
        let mut strides = SmallVec::<[usize; 7]>::new();
        for &dim in dim_order.iter() {
            dims.push(self.dims[dim]);
            strides.push(self.strides[dim]);
        }
        Self::new(&dims, &strides)
    }

    // TODO: poor man's .view(), except it won't tell you if you mess up!
    pub fn shape_cast(&self, dims: &[isize]) -> Self {
        let mut mut_dims: SmallVec<[isize; 7]> = dims.into();
        let mut strides = SmallVec::<[usize; 7]>::new();
        let mut has_neg_dim = false;
        for dim in mut_dims.iter_mut() {
            if *dim < 0 {
                assert!(!has_neg_dim);
                has_neg_dim = true;
                *dim = self.dims.iter().product::<usize>() as isize
                    / dims.iter().filter(|&&x| x >= 0).product::<isize>();
            }
        }
        assert_eq!(
            mut_dims.iter().product::<isize>(),
            self.dims.iter().product::<usize>() as isize
        );
        let mut stride = 1;
        for &dim in mut_dims.iter().rev() {
            strides.push(stride as usize);
            stride *= dim;
        }
        strides.reverse();
        Self { dims: mut_dims.iter().map(|x| *x as usize).collect(), strides }
    }

    pub fn size(&self, dim: isize) -> usize {
        if dim < 0 {
            self.dims[self.dims.len() - (-dim as usize)]
        } else {
            self.dims[dim as usize]
        }
    }

    pub fn stride(&self, dim: isize) -> usize {
        if dim < 0 {
            self.strides[self.strides.len() - (-dim as usize)]
        } else {
            self.strides[dim as usize]
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum TensorStoragePtr<T> {
    Cpu(*const T),
    Hip(simt_hip_sys::hipDeviceptr_t),
}

impl<T> Copy for TensorStoragePtr<T> {}
impl<T> Clone for TensorStoragePtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum TensorStoragePtrMut<T> {
    Cpu(*mut T),
    Hip(simt_hip_sys::hipDeviceptr_t),
}

impl<T> Copy for TensorStoragePtrMut<T> {}
impl<T> Clone for TensorStoragePtrMut<T> {
    fn clone(&self) -> Self {
        *self
    }
}

pub enum TensorStorage<T> {
    Cpu(Vec<T>),
    Hip(HipBuffer),
}

pub struct Tensor<T> {
    storage: TensorStorage<T>,
    layout: TensorLayout,
    _dead: PhantomData<T>,
}

impl<T> Tensor<T> {
    pub fn as_view(&self) -> TensorView<T> {
        TensorView {
            ptr: match &self.storage {
                TensorStorage::Cpu(v) => TensorStoragePtr::Cpu(v.as_ptr()),
                TensorStorage::Hip(b) => TensorStoragePtr::Hip(b.ptr),
            },
            layout: self.layout.clone(),
            _dead: PhantomData,
        }
    }

    pub fn as_view_mut(&mut self) -> TensorViewMut<T> {
        TensorViewMut {
            ptr: match &mut self.storage {
                TensorStorage::Cpu(v) => TensorStoragePtrMut::Cpu(v.as_mut_ptr()),
                TensorStorage::Hip(b) => TensorStoragePtrMut::Hip(b.ptr),
            },
            layout: self.layout.clone(),
            _dead: PhantomData,
        }
    }
}

impl<T: Copy + Default> Tensor<T> {
    pub fn new_cpu(dims: &[usize]) -> Self {
        let layout = TensorLayout::row_major(dims);
        let storage = TensorStorage::Cpu(vec![T::default(); layout.largest_address() + 1]);
        Tensor { storage, layout, _dead: PhantomData }
    }

    pub fn new_hip(dims: &[usize]) -> anyhow::Result<Self> {
        Self::new_hip_layout(TensorLayout::row_major(dims))
    }

    pub fn new_hip_layout(layout: TensorLayout) -> anyhow::Result<Self> {
        let buf = HipBuffer::new((layout.largest_address() + 1) * std::mem::size_of::<T>())?;
        unsafe {
            simt_hip::hip_call(|| {
                simt_hip_sys::library().hipMemset(buf.ptr as *mut _, 0, buf.size)
            })?;
        }
        Ok(Tensor { storage: TensorStorage::Hip(buf), layout, _dead: PhantomData })
    }

    pub fn from_vec(vec: Vec<T>, layout: TensorLayout) -> Self {
        assert!(vec.len() > layout.largest_address());
        Tensor { storage: TensorStorage::Cpu(vec), layout, _dead: PhantomData }
    }

    pub fn into_hip(self) -> anyhow::Result<Self> {
        match self.storage {
            TensorStorage::Cpu(v) => {
                let mut buf = HipBuffer::new(v.len() * std::mem::size_of::<T>())?;
                buf.copy_from_slice(&v)?;
                Ok(Tensor {
                    storage: TensorStorage::Hip(buf),
                    layout: self.layout,
                    _dead: PhantomData,
                })
            }
            TensorStorage::Hip(_) => Ok(self),
        }
    }

    pub fn as_gpu_ptr(&self) -> *const T {
        match &self.storage {
            TensorStorage::Cpu(_) => panic!("not a gpu tensor"),
            TensorStorage::Hip(b) => b.ptr as *const T,
        }
    }

    pub fn as_mut_gpu_ptr(&mut self) -> *mut T {
        match &self.storage {
            TensorStorage::Cpu(_) => panic!("not a gpu tensor"),
            TensorStorage::Hip(b) => b.ptr as *mut T,
        }
    }

    pub fn size(&self, dim: isize) -> usize {
        self.layout.size(dim)
    }

    pub fn stride(&self, dim: isize) -> usize {
        self.layout.stride(dim)
    }
}

impl<T: Default + Debug + Copy> Debug for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(f)
    }
}

#[repr(C)]
pub struct TensorView<'a, T> {
    ptr: TensorStoragePtr<T>,
    layout: TensorLayout,
    _dead: PhantomData<&'a T>,
}

pub struct ElidingRangeIterator {
    end: usize,
    current: usize,
    threshold: usize,
    edge_items: usize,
}

impl ElidingRangeIterator {
    pub fn new(n: usize, elide_threshold: usize, edge_items: usize) -> Self {
        Self { end: n, current: 0, threshold: elide_threshold, edge_items }
    }
}

impl Iterator for ElidingRangeIterator {
    type Item = (bool, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let result = self.current;
            self.current += 1;

            if self.end > self.threshold {
                if self.current >= self.edge_items && self.current < self.end - self.edge_items {
                    self.current = self.end - self.edge_items;
                    return Some((true, result));
                }
            }

            Some((false, result))
        } else {
            None
        }
    }
}

fn format_slice_with_layout<T: Debug + Copy>(
    f: &mut Formatter<'_>, slice: &[T], dim: usize, layout: &TensorLayout,
) -> std::fmt::Result {
    let dims_right = layout.dims.len() - dim - 1;
    let mut first = true;
    if dims_right == 0 {
        for (skip, i) in ElidingRangeIterator::new(layout.dims[dim], 6, 3) {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            Debug::fmt(&slice[layout.strides[dim] * i], f)?;
            if skip {
                f.write_str(", ...")?;
            }
        }
    } else {
        for (skip, i) in ElidingRangeIterator::new(layout.dims[dim], 6, 3) {
            if !first {
                write!(f, "\n")?;
            }
            first = false;
            write!(f, "[")?;
            format_slice_with_layout(f, &slice[layout.strides[dim] * i..], dim + 1, layout)?;
            write!(f, "]")?;
            if skip {
                write!(f, "\n...")?;
            }
        }
    }
    Ok(())
}

impl<'a, T: Default + Debug + Copy> Debug for TensorView<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.ptr {
            TensorStoragePtr::Cpu(p) => {
                let slice =
                    unsafe { std::slice::from_raw_parts(p, self.layout.largest_address() + 1) };
                format_slice_with_layout(f, slice, 0, &self.layout)?;
            }
            TensorStoragePtr::Hip(b) => {
                let storage_size = self.layout.largest_address() + 1;
                let mut cpu_data = vec![T::default(); storage_size];
                unsafe {
                    simt_hip::hip_call(|| {
                        simt_hip_sys::library().hipMemcpy(
                            cpu_data.as_mut_ptr() as *mut std::ffi::c_void,
                            b as *const std::ffi::c_void,
                            storage_size * std::mem::size_of::<T>(),
                            simt_hip_sys::hipMemcpyKind::hipMemcpyDeviceToHost,
                        )
                    })
                    .map_err(|_| std::fmt::Error)?;
                }
                format_slice_with_layout(f, &cpu_data, 0, &self.layout)?;
            }
        }
        Ok(())
    }
}

impl<'a, T> TensorView<'a, T> {
    pub fn as_gpu_ptr(&self) -> *const T {
        match self.ptr {
            TensorStoragePtr::Cpu(_) => panic!("not a gpu tensor"),
            TensorStoragePtr::Hip(b) => b as *const T,
        }
    }

    pub fn size(&self, dim: isize) -> usize {
        self.layout.size(dim)
    }

    pub fn stride(&self, dim: isize) -> usize {
        self.layout.stride(dim)
    }

    pub fn permute(&self, dim_order: &[usize]) -> Self {
        Self { ptr: self.ptr, layout: self.layout.permute(dim_order), _dead: PhantomData }
    }

    pub fn shape_cast(&self, shape: &[isize]) -> Self {
        Self { ptr: self.ptr, layout: self.layout.shape_cast(shape), _dead: PhantomData }
    }
}

#[repr(C)]
pub struct TensorViewMut<'a, T> {
    ptr: TensorStoragePtrMut<T>,
    layout: TensorLayout,
    _dead: PhantomData<&'a mut T>,
}

impl<'a, T> TensorViewMut<'a, T> {
    pub fn as_view(&self) -> TensorView<'_, T> {
        TensorView {
            ptr: match &self.ptr {
                TensorStoragePtrMut::Cpu(p) => TensorStoragePtr::Cpu(*p),
                TensorStoragePtrMut::Hip(p) => TensorStoragePtr::Hip(*p),
            },
            layout: self.layout.clone(),
            _dead: PhantomData,
        }
    }

    pub fn as_gpu_ptr(&self) -> *const T {
        match self.ptr {
            TensorStoragePtrMut::Cpu(_) => panic!("not a gpu tensor"),
            TensorStoragePtrMut::Hip(b) => b as *const T,
        }
    }

    pub fn as_mut_gpu_ptr(&mut self) -> *mut T {
        match self.ptr {
            TensorStoragePtrMut::Cpu(_) => panic!("not a gpu tensor"),
            TensorStoragePtrMut::Hip(b) => b as *mut T,
        }
    }

    pub fn size(&self, dim: isize) -> usize {
        self.layout.size(dim)
    }

    pub fn stride(&self, dim: isize) -> usize {
        self.layout.stride(dim)
    }

    pub fn permute(&mut self, dim_order: &[usize]) -> Self {
        Self { ptr: self.ptr, layout: self.layout.permute(dim_order), _dead: PhantomData }
    }

    pub fn shape_cast(&mut self, shape: &[isize]) -> Self {
        Self { ptr: self.ptr, layout: self.layout.shape_cast(shape), _dead: PhantomData }
    }
}

impl<'a, T: Display + Default + Debug + Copy> Debug for TensorViewMut<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(f)
    }
}

pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_N_MELS: usize = 80;
pub const WHISPER_HOP_LENGTH: usize = 160;
pub const WHISPER_CHUNK_LENGTH: usize = 30;
pub const WHISPER_CHUNK_FRAMES: usize =
    WHISPER_CHUNK_LENGTH * WHISPER_SAMPLE_RATE as usize / WHISPER_HOP_LENGTH;

#[repr(u32)]
enum Conv1dActivation {
    None = 0,
    GELU = 1,
}

#[repr(u32)]
enum BinaryOp {
    ADD = 1,
}

#[repr(u32)]
enum MatmulLoadOp {
    IDENTITY = 0,
    SCALE = 1,
}

enum MatmulLoad {
    Identity,
    Scale(f32),
}

impl MatmulLoad {
    pub fn lower(&self) -> MatmulLoadOp {
        match self {
            MatmulLoad::Identity => MatmulLoadOp::IDENTITY,
            MatmulLoad::Scale(_) => MatmulLoadOp::SCALE,
        }
    }
}

#[repr(u32)]
enum MatmulStoreOp {
    IDENTITY = 0,
    GELU_BIAS = 1,
    BETA_GELU_BIAS = 2,
    BETA_BIAS = 3,
}

enum MatmulStore<'a> {
    Identity,
    GeluBias(TensorView<'a, f16>),
    BetaGeluBias(f32, TensorView<'a, f16>),
    BetaBias(f32, TensorView<'a, f16>),
}

impl<'a> MatmulStore<'a> {
    pub fn lower(&self) -> MatmulStoreOp {
        match self {
            MatmulStore::Identity => MatmulStoreOp::IDENTITY,
            MatmulStore::GeluBias(_) => MatmulStoreOp::GELU_BIAS,
            MatmulStore::BetaGeluBias(_, _) => MatmulStoreOp::BETA_GELU_BIAS,
            MatmulStore::BetaBias(_, _) => MatmulStoreOp::BETA_BIAS,
        }
    }
}

struct WhisperKernels {
    conv1d: Kernel<(
        *mut f16,
        *const f16,
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
    )>,
    layer_norm:
        Kernel<(*mut f16, *const f16, *const f16, *const f16, i32, i32, i32, i32, i32, i32, f32)>,
    linear: Kernel<(
        *mut f16,
        *const f16,
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
        f32,
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
    )>,
    elementwise_binary_2d_f16:
        Kernel<(*mut f16, *const f16, *const f16, i32, i32, i32, i32, i32, i32, i32, i32, u32)>,
    softmax_rows: Kernel<(*mut f16, *const f16, i32, i32, f32)>,
    embed: Kernel<(*mut f16, *const i32, i32, i32, *const f16)>,
}

impl WhisperKernels {
    pub fn linear(
        &self, mut output: TensorViewMut<f16>, left: TensorView<f16>, right: TensorView<f16>,
        bias: Option<TensorView<f16>>, beta: f32,
    ) -> anyhow::Result<()> {
        self.linear.launch(
            LaunchParams {
                blocks: (
                    ceil_div(output.size(-1) as u64, 16) as u32,
                    ceil_div(output.size(-2) as u64, 16) as u32,
                    1,
                ),
                threads: (16, 16, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                output.as_mut_gpu_ptr(),
                left.as_gpu_ptr(),
                right.as_gpu_ptr(),
                bias.map(|b| b.as_gpu_ptr()).unwrap_or(std::ptr::null()),
                left.size(-2) as i32,
                left.size(-1) as i32,
                right.size(-1) as i32,
                output.stride(-1) as i32,
                output.stride(-2) as i32,
                left.stride(-1) as i32,
                left.stride(-2) as i32,
                right.stride(-1) as i32,
                right.stride(-2) as i32,
                beta,
            ),
        )?;
        Ok(())
    }

    pub fn layer_norm(
        &self, output: &mut TensorViewMut<f16>, input: TensorView<f16>, weight: TensorView<f16>,
        bias: TensorView<f16>, eps: f32,
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

    pub fn matmul_f16(
        &self, output: &mut TensorViewMut<f16>, left: TensorView<f16>, right: TensorView<f16>,
        load_op: MatmulLoad, store_op: MatmulStore,
    ) -> anyhow::Result<()> {
        let bias = match &store_op {
            MatmulStore::Identity => None,
            MatmulStore::GeluBias(bias) => Some(bias),
            MatmulStore::BetaGeluBias(_, bias) => Some(bias),
            MatmulStore::BetaBias(_, bias) => Some(bias),
        };
        let scale = match &load_op {
            MatmulLoad::Identity => 1.0,
            MatmulLoad::Scale(scale) => *scale,
        };
        let beta = match &store_op {
            MatmulStore::BetaGeluBias(beta, _) => *beta,
            MatmulStore::BetaBias(beta, _) => *beta,
            _ => 0.0,
        };
        let bias_ptr = bias.map(|b| b.as_gpu_ptr()).unwrap_or(std::ptr::null());
        self.matmul_f16.launch(
            LaunchParams {
                blocks: (
                    ceil_div(output.size(-1) as u64, 16) as u32,
                    ceil_div(output.size(-2) as u64, 16) as u32,
                    if output.layout.dims.len() > 2 { output.size(-3) as u32 } else { 1 },
                ),
                threads: (16, 16, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                output.as_mut_gpu_ptr(),
                left.as_gpu_ptr(),
                right.as_gpu_ptr(),
                if output.layout.dims.len() > 2 { output.size(-3) as i32 } else { 1 },
                left.size(-2) as i32,
                left.size(-1) as i32,
                right.size(-1) as i32,
                output.stride(-1) as i32,
                output.stride(-2) as i32,
                if output.layout.dims.len() > 2 { output.stride(-3) } else { 0 } as i32,
                left.stride(-1) as i32,
                left.stride(-2) as i32,
                if left.layout.dims.len() > 2 { left.stride(-3) } else { 0 } as i32,
                right.stride(-1) as i32,
                right.stride(-2) as i32,
                if right.layout.dims.len() > 2 { right.stride(-3) } else { 0 } as i32,
                bias_ptr,
                beta,
                scale,
                store_op.lower() as u32,
                load_op.lower() as u32,
            ),
        )?;
        Ok(())
    }

    pub fn softmax_rows(
        &self, output: TensorViewMut<f16>, input: TensorView<f16>, temperature: f32,
    ) -> anyhow::Result<()> {
        todo!()
        //
    }

    pub fn softmax_rows_inplace(
        &self, mut output: TensorViewMut<f16>, temperature: f32,
    ) -> anyhow::Result<()> {
        self.softmax_rows.launch(
            LaunchParams {
                blocks: (
                    output.size(-2) as u32,
                    if output.layout.dims.len() > 2 { output.size(-3) as u32 } else { 1 },
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

    pub fn embed(
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

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct WhisperDims {
    pub n_mels: i32,
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
}

#[derive(Deserialize)]
pub struct WhisperModelState {
    dims: WhisperDims,
    model_state_dict: serde_pickle::Value,
}

impl<'de> pickle::ModelState<'de> for WhisperModelState {
    type Metadata = WhisperDims;
    type LoadParams = ();
    fn state_dict(&self) -> &serde_pickle::Value {
        &self.model_state_dict
    }
    fn into_metadata(self) -> Self::Metadata {
        self.dims
    }
}

#[derive(Debug)]
pub struct WhisperLayerNorm {
    weight: Tensor<f16>,
    bias: Tensor<f16>,
}

impl WhisperLayerNorm {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: load_tensor(pickle, &format!("{}.weight", prefix))?,
            bias: load_tensor(pickle, &format!("{}.bias", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperAttention {
    query: WhisperLinear,
    key: Tensor<f16>,
    value: WhisperLinear,
    out: WhisperLinear,
}

impl WhisperAttention {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            query: WhisperLinear::new(pickle, &format!("{}.query", prefix))?,
            key: load_tensor(pickle, &format!("{}.key.weight", prefix))?,
            value: WhisperLinear::new(pickle, &format!("{}.value", prefix))?,
            out: WhisperLinear::new(pickle, &format!("{}.out", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperLinear {
    weight: Tensor<f16>,
    bias: Tensor<f16>,
}

impl WhisperLinear {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: load_tensor(pickle, &format!("{}.weight", prefix))?,
            bias: load_tensor(pickle, &format!("{}.bias", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperTransformerBlock {
    attn: WhisperAttention,
    cross_attn: Option<WhisperAttention>,
    attn_ln: WhisperLayerNorm,
    mlp_0: WhisperLinear,
    mlp_2: WhisperLinear,
    mlp_ln: WhisperLayerNorm,
}

impl WhisperTransformerBlock {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            attn: WhisperAttention::new(pickle, &format!("{}.attn", prefix))?,
            cross_attn: None,
            attn_ln: WhisperLayerNorm::new(pickle, &format!("{}.attn_ln", prefix))?,
            mlp_0: WhisperLinear::new(pickle, &format!("{}.mlp.0", prefix))?,
            mlp_2: WhisperLinear::new(pickle, &format!("{}.mlp.2", prefix))?,
            mlp_ln: WhisperLayerNorm::new(pickle, &format!("{}.mlp_ln", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperConv1d {
    weight: Tensor<f16>,
    bias: Tensor<f16>,
}

impl WhisperConv1d {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: load_tensor(pickle, &format!("{}.weight", prefix))?,
            bias: load_tensor(pickle, &format!("{}.bias", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperAudioEncoder {
    conv1: WhisperConv1d,
    conv2: WhisperConv1d,
    position_embedding: Tensor<f16>,
    layers: Vec<WhisperTransformerBlock>,
    ln_post: WhisperLayerNorm,
}

impl WhisperAudioEncoder {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            conv1: WhisperConv1d::new(pickle, &format!("{}.conv1", prefix))?,
            conv2: WhisperConv1d::new(pickle, &format!("{}.conv2", prefix))?,
            position_embedding: sinusoid_position_embedding(&pickle.metadata).into_hip()?,
            layers: (0..pickle.metadata.n_audio_layer)
                .map(|i| WhisperTransformerBlock::new(pickle, &format!("{}.blocks.{}", prefix, i)))
                .collect::<anyhow::Result<Vec<_>>>()?,
            ln_post: WhisperLayerNorm::new(pickle, &format!("{}.ln_post", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperTextDecoder {
    token_embedding: Tensor<f16>,
    positional_embedding: Tensor<f16>,
    ln: WhisperLayerNorm,
    layers: Vec<WhisperTransformerBlock>,
}

impl WhisperTextDecoder {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            token_embedding: load_tensor(pickle, &format!("{}.token_embedding.weight", prefix))?,
            positional_embedding: load_tensor(pickle, &format!("{}.positional_embedding", prefix))?,
            ln: WhisperLayerNorm::new(pickle, &format!("{}.ln", prefix))?,
            layers: (0..pickle.metadata.n_text_layer)
                .map(|i| WhisperTransformerBlock::new(pickle, &format!("{}.blocks.{}", prefix, i)))
                .collect::<anyhow::Result<Vec<_>>>()?,
        })
    }
}

pub struct WhisperModel {
    dims: WhisperDims,
    encoder: WhisperAudioEncoder,
    decoder: WhisperTextDecoder,
    tokenizer: CoreBPE,
}

impl Debug for WhisperModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("WhisperModel")
            .field("dims", &self.dims)
            .field("encoder", &self.encoder)
            .field("decoder", &self.decoder)
            .finish()
    }
}

impl WhisperModel {
    pub fn new(pickle: &PickledModel<WhisperDims>) -> anyhow::Result<Self> {
        Ok(Self {
            dims: pickle.metadata.clone(),
            encoder: WhisperAudioEncoder::new(pickle, "encoder")?,
            decoder: WhisperTextDecoder::new(pickle, "decoder")?,
            tokenizer: if pickle.metadata.n_vocab == 51865 {
                tiktoken_rs::whisper_multilingual()?
            } else {
                tiktoken_rs::whisper_gpt2()?
            },
        })
    }
}

fn sinusoid_position_embedding(dims: &WhisperDims) -> Tensor<f16> {
    let mut pos_embedding_vec =
        vec![f16::from_f32(0.0); dims.n_audio_ctx as usize * dims.n_audio_state as usize];
    let increment = (10000.0f32).ln() / (dims.n_audio_state / 2 - 1) as f32;
    for i in 0..dims.n_audio_ctx as usize {
        for j in 0..dims.n_audio_state as usize / 2 {
            let theta = i as f32 * (j as f32 * -increment).exp();
            pos_embedding_vec[i * dims.n_audio_state as usize + j] = f16::from_f32(theta.sin());
            pos_embedding_vec
                [i * dims.n_audio_state as usize + dims.n_audio_state as usize / 2 + j] =
                f16::from_f32(theta.cos());
        }
    }
    let pos_embedding = Tensor::from_vec(
        pos_embedding_vec,
        TensorLayout::row_major(&[dims.n_audio_ctx as usize, dims.n_audio_state as usize]),
    );
    pos_embedding
}

struct WhisperContextCacheLayer {
    key: Tensor<f16>,
    value: Tensor<f16>,
}

struct WhisperContext {
    model: Arc<WhisperModel>,
    kernels: Arc<WhisperKernels>,
    mel_transform: mel::LogMelSpectrogramTransform,
    kv_cache: Vec<WhisperContextCacheLayer>,
}

impl WhisperContext {
    pub fn new(model: Arc<WhisperModel>, kernels: Arc<WhisperKernels>) -> anyhow::Result<Self> {
        Ok(Self {
            kernels,
            mel_transform: mel::LogMelSpectrogramTransform::new(
                WHISPER_N_FFT,
                WHISPER_N_MELS,
                WHISPER_HOP_LENGTH,
                WHISPER_SAMPLE_RATE as f32,
            ),
            kv_cache: (0..model.dims.n_text_layer)
                .map(|_| {
                    Ok(WhisperContextCacheLayer {
                        key: Tensor::new_hip(&[
                            model.dims.n_text_ctx as usize,
                            model.dims.n_text_state as usize,
                        ])?,
                        value: Tensor::new_hip(&[
                            model.dims.n_text_ctx as usize,
                            model.dims.n_text_state as usize,
                        ])?,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?,
            model,
        })
    }

    fn process_layer(
        &self, layer: &WhisperTransformerBlock, hidden_state: &mut TensorViewMut<f16>,
    ) -> anyhow::Result<()> {
        let heads = self.model.dims.n_audio_head as isize;
        let mut ln_out = Tensor::new_hip(&hidden_state.layout.dims)?;
        let mut query = Tensor::new_hip(&ln_out.layout.dims)?;
        let mut key = Tensor::new_hip(&ln_out.layout.dims)?;
        let mut value = Tensor::new_hip(&ln_out.layout.dims)?;
        let mut mlp_hidden = Tensor::new_hip(&[
            self.model.dims.n_audio_ctx as usize,
            self.model.dims.n_audio_state as usize * 4,
        ])?;
        let mut qkv = Tensor::new_hip(&[
            self.model.dims.n_audio_ctx as usize,
            self.model.dims.n_audio_state as usize,
        ])?;
        self.kernels.layer_norm(
            &mut ln_out.as_view_mut(),
            hidden_state.as_view(),
            layer.attn_ln.weight.as_view(),
            layer.attn_ln.bias.as_view(),
            1.0e-5,
        )?;
        self.kernels.linear(
            query.as_view_mut(),
            ln_out.as_view(),
            layer.attn.query.weight.as_view(),
            Some(layer.attn.query.bias.as_view()),
            0.0,
        )?;
        self.kernels.linear(
            key.as_view_mut(),
            ln_out.as_view(),
            layer.attn.key.as_view(),
            None,
            0.0,
        )?;
        self.kernels.linear(
            value.as_view_mut(),
            ln_out.as_view(),
            layer.attn.value.weight.as_view(),
            Some(layer.attn.value.bias.as_view()),
            0.0,
        )?;
        let q_view =
            query.as_view().shape_cast(&[query.size(-2) as isize, heads, -1]).permute(&[1, 0, 2]);
        let k_view =
            key.as_view().shape_cast(&[key.size(-2) as isize, heads, -1]).permute(&[1, 2, 0]);
        let v_view =
            value.as_view().shape_cast(&[value.size(-2) as isize, heads, -1]).permute(&[1, 0, 2]);
        let mut qk = Tensor::new_hip(&[heads as usize, q_view.size(-2), k_view.size(-1)])?;
        self.kernels.matmul_f16(
            &mut qk.as_view_mut(),
            q_view,
            k_view,
            MatmulLoad::Scale(
                (self.model.dims.n_audio_state as f32 / self.model.dims.n_audio_head as f32)
                    .powf(-0.25),
            ),
            MatmulStore::Identity,
        )?;
        self.kernels.softmax_rows_inplace(qk.as_view_mut(), 1.0)?;
        let mut qkv_view =
            qkv.as_view_mut().shape_cast(&[value.size(-2) as isize, heads, -1]).permute(&[1, 0, 2]);
        self.kernels.matmul_f16(
            &mut qkv_view,
            qk.as_view(),
            v_view,
            MatmulLoad::Identity,
            MatmulStore::Identity,
        )?;
        self.kernels.matmul_f16(
            hidden_state,
            qkv.as_view(),
            layer.attn.out.weight.as_view().permute(&[0, 1, 3, 2]),
            MatmulLoad::Identity,
            MatmulStore::BetaBias(1.0, layer.attn.out.bias.as_view()),
        )?;
        self.kernels.layer_norm(
            &mut ln_out.as_view_mut(),
            hidden_state.as_view(),
            layer.mlp_ln.weight.as_view(),
            layer.mlp_ln.bias.as_view(),
            1.0e-5,
        )?;
        self.kernels.matmul_f16(
            &mut mlp_hidden.as_view_mut(),
            ln_out.as_view(),
            layer.mlp_0.weight.as_view().permute(&[0, 1, 3, 2]),
            MatmulLoad::Identity,
            MatmulStore::BetaGeluBias(0.0, layer.mlp_0.bias.as_view()),
        )?;
        self.kernels.matmul_f16(
            hidden_state,
            mlp_hidden.as_view(),
            layer.mlp_2.weight.as_view().permute(&[0, 1, 3, 2]),
            MatmulLoad::Identity,
            MatmulStore::BetaBias(1.0, layer.mlp_2.bias.as_view()),
        )?;
        Ok(())
    }

    pub fn encode(&mut self, wave: &[f32]) -> anyhow::Result<Tensor<f16>> {
        // TODO we should be accepting a fixed length and not allocate these each time
        let mut complex_scratch =
            vec![Complex32::new(0.0, 0.0); self.mel_transform.complex_scratch_size(wave.len())];
        let mut real_scratch = vec![0.0; self.mel_transform.real_scratch_size(wave.len())];
        let mut mel_spec = vec![0.0; self.mel_transform.output_size(wave.len())];
        self.mel_transform.process(&mut mel_spec, wave, &mut complex_scratch, &mut real_scratch);
        let mut mels_half = vec![f16::from_f32(0.0); mel_spec.len()];
        for (l, r) in mels_half.iter_mut().zip(mel_spec.iter()) {
            *l = f16::from_f32(*r);
        }
        let mels_half = Tensor::from_vec(
            mels_half,
            TensorLayout::row_major(&[WHISPER_N_MELS, self.mel_transform.num_cols(wave.len())]),
        )
        .into_hip()?;
        let mut conv_out =
            Tensor::new_hip(&[self.model.dims.n_audio_state as usize, mels_half.size(-1)])?;
        let conv_params = LaunchParams {
            blocks: (
                ceil_div(mels_half.size(-1) as u64, 16) as u32,
                ceil_div(self.model.dims.n_audio_state as u64, 16) as u32,
                1,
            ),
            threads: (16, 16, 1),
            shared_mem: 0,
            stream: None,
        };
        self.kernels.conv1d.launch(
            conv_params.clone(),
            (
                conv_out.as_mut_gpu_ptr(),
                mels_half.as_gpu_ptr(),
                self.model.encoder.conv1.weight.as_gpu_ptr(),
                self.model.encoder.conv1.bias.as_gpu_ptr(),
                WHISPER_N_MELS as i32,
                self.model.dims.n_audio_state,
                3,
                mels_half.size(-1) as i32,
                mels_half.size(-1) as i32,
                1,
                1,
                Conv1dActivation::GELU as i32,
            ),
        )?;
        let mut hidden_state =
            Tensor::new_hip(&[self.model.dims.n_audio_state as usize, mels_half.size(-1) / 2])?;
        self.kernels.conv1d.launch(
            conv_params,
            (
                hidden_state.as_mut_gpu_ptr(),
                conv_out.as_gpu_ptr(),
                self.model.encoder.conv2.weight.as_gpu_ptr(),
                self.model.encoder.conv2.bias.as_gpu_ptr(),
                self.model.dims.n_audio_state,
                self.model.dims.n_audio_state,
                3,
                mels_half.size(-1) as i32,
                mels_half.size(-1) as i32 / 2,
                2,
                1,
                Conv1dActivation::GELU as i32,
            ),
        )?;
        let mut hidden_state = hidden_state.as_view_mut().permute(&[1, 0]);
        self.kernels.elementwise_binary_2d_f16.launch(
            LaunchParams {
                blocks: (
                    ceil_div(hidden_state.size(-1) as u64, 16) as u32,
                    ceil_div(hidden_state.size(-2) as u64, 16) as u32,
                    1,
                ),
                threads: (16, 16, 1),
                shared_mem: 0,
                stream: None,
            },
            (
                hidden_state.as_mut_gpu_ptr(),
                hidden_state.as_gpu_ptr(),
                self.model.encoder.position_embedding.as_gpu_ptr(),
                hidden_state.size(-1) as i32,
                hidden_state.size(-2) as i32,
                hidden_state.stride(-1) as i32,
                hidden_state.stride(-2) as i32,
                hidden_state.stride(-1) as i32,
                hidden_state.stride(-2) as i32,
                self.model.encoder.position_embedding.stride(-1) as i32,
                self.model.encoder.position_embedding.stride(-2) as i32,
                BinaryOp::ADD as u32,
            ),
        )?;
        for layer in &self.model.encoder.layers {
            self.process_layer(layer, &mut hidden_state)?;
        }
        let mut features = Tensor::new_hip(&[
            self.model.dims.n_audio_ctx as usize,
            self.model.dims.n_audio_state as usize,
        ])?;
        self.kernels.layer_norm(
            &mut features.as_view_mut(),
            hidden_state.as_view(),
            self.model.encoder.ln_post.weight.as_view(),
            self.model.encoder.ln_post.bias.as_view(),
            1.0e-5,
        )?;
        //println!("features {:?}\n{:>+7.4?}", features.layout, features);
        Ok(features)
    }

    // TODO(eiz): just sample tokens for now until irl decode logic is written
    // need to figure out a proper streaming solution anyway
    pub fn decode(
        &mut self, features: TensorView<f16>, tokens: &[i32],
    ) -> anyhow::Result<Tensor<f16>> {
        let mut embedded = Tensor::new_hip(&[tokens.len(), self.model.dims.n_text_state as usize])?;
        let tokens_gpu =
            Tensor::from_vec(tokens.into(), TensorLayout::row_major(&[tokens.len()])).into_hip()?;
        self.kernels.embed(
            &mut embedded.as_view_mut(),
            tokens_gpu.as_view(),
            self.model.decoder.token_embedding.as_view(),
        )?;
        todo!()
        //
    }
}

fn wav2float_mono(data: &wav::BitDepth) -> Vec<f32> {
    match data {
        wav::BitDepth::Eight(v) => v.iter().map(|x| *x as f32 / 128.0 - 1.0).collect(),
        wav::BitDepth::Sixteen(v) => v.iter().map(|x| *x as f32 / 32768.0).collect(),
        wav::BitDepth::TwentyFour(v) => v.iter().map(|x| *x as f32 / 8388608.0).collect(),
        wav::BitDepth::ThirtyTwoFloat(v) => v.iter().map(|x| *x as f32).collect(),
        wav::BitDepth::Empty => vec![],
    }
}

fn load_tensor<T>(pickled: &PickledModel<T>, name: &str) -> anyhow::Result<Tensor<f16>> {
    let pickled_tensor =
        pickled.tensors.get(name).ok_or_else(|| anyhow::anyhow!("tensor {} not found", name))?;
    let mut tensor =
        Tensor::new_hip_layout(TensorLayout::new(&pickled_tensor.shape, &pickled_tensor.stride))?;
    match tensor.storage {
        TensorStorage::Hip(ref mut b) => {
            b.copy_from_slice(&pickled.mapping.data()[pickled_tensor.range.clone()])?;
        }
        _ => unreachable!(),
    }
    Ok(tensor)
}

fn wav_test<P: AsRef<Path>, Q: AsRef<Path>>(path: P, model_path: Q) -> anyhow::Result<()> {
    let phys = HipPhysicalDevice::get(0)?;
    let device = Arc::new(HipDevice::new(phys)?);
    let _scope = device.lock()?;
    // BIG TODO: loading each kernel as a separate module like this is super not ergonomic
    // use a better way
    let capability = phys.capability()?;
    let module_conv1d = HipModule::find(capability, adrastea_kernels::conv1d)?;
    let module_layer_norm = HipModule::find(capability, adrastea_kernels::layer_norm)?;
    let module_elementwise = HipModule::find(capability, adrastea_kernels::elementwise)?;
    let module_matmul = HipModule::find(capability, adrastea_kernels::matmul)?;
    let module_softmax_rows = HipModule::find(capability, adrastea_kernels::softmax_rows)?;
    let module_embed = HipModule::find(capability, adrastea_kernels::embed)?;
    let kernels = WhisperKernels {
        conv1d: Kernel::new(&module_conv1d, "conv1d")?,
        layer_norm: Kernel::new(&module_layer_norm, "layer_norm")?,
        linear: Kernel::new(&module_matmul, "linear")?,
        matmul_f16: Kernel::new(&module_matmul, "matmul_f16")?,
        elementwise_binary_2d_f16: Kernel::new(&module_elementwise, "elementwise_binary_2d_f16")?,
        softmax_rows: Kernel::new(&module_softmax_rows, "softmax_rows")?,
        embed: Kernel::new(&module_embed, "embed")?,
    };
    let model = WhisperModel::new(&WhisperModelState::load(model_path, ())?)?;
    let mut context = WhisperContext::new(Arc::new(model), Arc::new(kernels))?;
    let mut fp = File::open(path)?;
    // TODO: 'wav' eager loads everything =/
    let (header, data) = wav::read(&mut fp)?;
    println!("{:#?}", header);
    println!("{:#?}", context.model.dims);
    let data_len = match &data {
        wav::BitDepth::Eight(v) => v.len(),
        wav::BitDepth::Sixteen(v) => v.len(),
        wav::BitDepth::TwentyFour(v) => v.len(),
        wav::BitDepth::ThirtyTwoFloat(v) => v.len(),
        wav::BitDepth::Empty => 0,
    };
    println!("samples: {}", data_len / header.channel_count as usize);
    println!(
        "duration: {}s",
        data_len as f32 / (header.sampling_rate as f32 * header.channel_count as f32)
    );
    if header.sampling_rate != WHISPER_SAMPLE_RATE {
        bail!(
            "unsupported sample rate {} x{}, resample to 16khz mono",
            header.sampling_rate,
            header.channel_count
        );
    }
    let mut wave = wav2float_mono(&data);
    wave.extend(std::iter::repeat(0.0).take(WHISPER_SAMPLE_RATE as usize * WHISPER_CHUNK_LENGTH));
    let wave = &wave[0..WHISPER_SAMPLE_RATE as usize * WHISPER_CHUNK_LENGTH];
    let features = context.encode(wave)?;
    let tokens = (context.model.tokenizer)
        // language of glorious mother nation
        .encode_with_special_tokens("<|startoftranscript|><|en|><|transcribe|>")
        .iter()
        .map(|x| *x as i32)
        .collect::<Vec<_>>();
    println!("initial tokens {:?}", tokens);
    let _logits = context.decode(features.as_view(), &tokens)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    println!("The endless sea.");
    if args.len() >= 2 && args[1] == "cuda" {
        unsafe { cuda_square()? }
    } else if args.len() >= 2 && args[1] == "hip" {
        hip_square()?
    } else if args.len() >= 3 && args[1] == "load" {
        let dict_path = if args.len() >= 4 { Some(args[3].as_str()) } else { None };
        let model = PickledModel::load_file(&args[2], dict_path)?;
        println!("{:#?}", model.tensors);
    } else if args.len() >= 3 && args[1] == "load_whisper" {
        let model = PickledModel::load_typed::<WhisperModelState, _>(&args[2], ())?;
        println!("{:#?}", model.tensors);
        println!("{:#?}", model.metadata);
    } else if args.len() >= 4 && args[1] == "wav" {
        wav_test(&args[2], &args[3])?;
    } else if args.len() >= 2 && args[1] == "vulkan" {
        unsafe { vulkan_square()? }
    } else {
        println!("test commands: cuda, hip, load, wav, vulkan");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{ElidingRangeIterator, Tensor, TensorLayout};

    #[test]
    fn test_print_tensor() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], TensorLayout::row_major(&[2, 2]));
        println!("{:?}", tensor);
        let tensor = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            TensorLayout::row_major(&[3, 3]),
        );
        println!("standard\n{:?}\n", tensor);
        println!("transpose\n{:?}", tensor.as_view().permute(&[1, 0]));
    }

    #[test]
    fn test_elided_range() {
        let mut indices = vec![];
        for (_skip, i) in ElidingRangeIterator::new(10, 6, 3) {
            indices.push(i);
        }
        assert_eq!(indices, vec![0, 1, 2, 7, 8, 9]);
    }

    #[test]
    fn shape_cast() {
        let tensor = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            TensorLayout::row_major(&[4, 4]),
        );

        println!("{:?}", tensor);
        println!("");
        println!("{:>5?}", tensor.as_view().shape_cast(&[-1, 8]));
    }
}
