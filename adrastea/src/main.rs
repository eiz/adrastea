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

use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{c_void, CStr},
    sync::Arc,
};

use anyhow::bail;
use ash::{vk, Entry};
use serde::{Deserialize, Serialize};
use simt_hip::{
    HipBuffer, HipDevice, HipModule, HipPhysicalDevice, HipStream, Kernel, LaunchParams,
};

pub mod pickle;

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
    reqs: &vk::MemoryRequirements,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
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
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 3, 0),
        ..Default::default()
    };
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
    let mut enabled_features_13 = vk::PhysicalDeviceVulkan13Features::builder()
        .synchronization2(true)
        .build();
    let mut enabled_features_12 = vk::PhysicalDeviceVulkan12Features::builder()
        .buffer_device_address(true)
        .build();
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
    dev.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo {
            ..Default::default()
        },
    )?;
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
        Ok(Self {
            cuda,
            device,
            context,
        })
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
        &mut self,
        src: *const std::ffi::c_void,
        size: usize,
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
        cuda.cuModuleGetFunction(
            x,
            module.inner,
            b"square_fp32_16x16\0".as_ptr() as *const i8,
        )
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
    let module = HipModule::find(capability, adrastea_kernels::square_fp32_16x16)?;
    let kernels = TestKernels {
        square_fp32_16x16: Kernel::new(&module, "square_fp32_16x16")?,
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
        (
            buf.ptr as *mut f32,
            buf.ptr as *mut f32,
            COLS as u32,
            ROWS as u32,
        ),
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

fn main() -> anyhow::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    println!("The endless sea.");
    if args.len() >= 2 && args[1] == "cuda" {
        unsafe { cuda_square()? }
    } else if args.len() >= 2 && args[1] == "hip" {
        hip_square()?
    } else if args.len() >= 3 && args[1] == "load" {
        let dict_path = if args.len() >= 4 {
            Some(args[3].as_str())
        } else {
            None
        };
        let model = pickle::PickledModel::load_file(&args[2], dict_path)?;
        println!("{:#?}", model.tensors);
    } else {
        unsafe { vulkan_square()? }
    }
    Ok(())
}
