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

use std::ffi::CStr;

use ash::{vk, Entry};

const THEM_SHADERS: &[u8] = include_bytes!("../shaders/square.comp.spv");

/*
mod simt {
    pub mod intrinsics {
        extern "C" {
            pub fn block_idx() -> (u32, u32, u32);
            pub fn block_dim() -> (u32, u32, u32);
            pub fn thread_idx() -> (u32, u32, u32);
            pub fn subgroup_size() -> u32;
        }
    }
}

struct half(u16);

#[simt::device]
unsafe fn foo(lhs: *const half, rhs: *const half, m: u32, n: u32, k: u32, x: u32, y: u32) -> f32 {
    0.0
}

#[simt::device]
#[inline(always)]
unsafe fn elementwise_2d() -> (u32, u32) {
    let block_idx = simt::intrinsics::block_idx();
    let block_dim = simt::intrinsics::block_dim();
    let thread_idx = simt::intrinsics::thread_idx();
    (
        block_idx.0 * block_dim.0 + thread_idx.0,
        block_idx.1 * block_dim.1 + thread_idx.1,
    )
}

#[simt::kernel]
unsafe fn dadada(out: *mut f32, lhs: *const half, rhs: *const half, m: u32, n: u32, k: u32) {
    let (x, y) = elementwise_2d();
    let data = foo(lhs, rhs, m, n, k, x, y);
    *(out.offset((y * n + x) as isize)) = data * 0.5;
}*/

fn ceil_div(a: vk::DeviceSize, b: vk::DeviceSize) -> vk::DeviceSize {
    (a + b - 1) / b
}

unsafe fn find_compute_queue_family(instance: &ash::Instance, phys_dev: vk::PhysicalDevice) -> u32 {
    let props = instance.get_physical_device_queue_family_properties(phys_dev);

    for (i, fam) in props.iter().enumerate() {
        if fam.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && fam.queue_flags.contains(vk::QueueFlags::TRANSFER)
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
struct WidthHeight {
    height: u32,
    width: u32,
}

unsafe fn main_unsafe() {
    const ROWS: vk::DeviceSize = 8;
    const COLS: vk::DeviceSize = 8;
    println!("The endless sea.");
    let entry = Entry::load().expect("failed to load vulkan");
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 3, 0),
        ..Default::default()
    };
    let instance = entry
        .create_instance(
            &vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&[b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8]),
            None,
        )
        .expect("derp");
    dbg!(instance.handle());
    let phys_devs = instance.enumerate_physical_devices().expect("derp");
    dbg!(&phys_devs);
    let queue_family_index = find_compute_queue_family(&instance, phys_devs[0]);
    let queue_infos = [vk::DeviceQueueCreateInfo {
        queue_count: 1,
        queue_family_index,
        ..Default::default()
    }];
    let mut create_info = vk::DeviceCreateInfo {
        queue_create_info_count: 1,
        p_queue_create_infos: queue_infos.as_ptr(),
        ..Default::default()
    };
    let mut features_13: vk::PhysicalDeviceVulkan13Features = Default::default();
    let mut features = vk::PhysicalDeviceFeatures2::builder()
        .push_next(&mut features_13)
        .build();
    instance.get_physical_device_features2(phys_devs[0], &mut features);
    create_info.p_next = &features as *const _ as *mut _;
    let dev = instance
        .create_device(phys_devs[0], &create_info, None)
        .expect("derp");
    dbg!(dev.handle());
    let fence = dev.create_fence(&Default::default(), None).expect("derp");
    let queue = dev.get_device_queue(queue_family_index, 0);
    dbg!(&queue);
    let command_pool = dev
        .create_command_pool(
            &vk::CommandPoolCreateInfo {
                queue_family_index,
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            },
            None,
        )
        .expect("derp");
    dbg!(&command_pool);
    let bufs = dev
        .allocate_command_buffers(&vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        })
        .expect("derp");
    dbg!(&bufs);
    let shader = dev
        .create_shader_module(
            &vk::ShaderModuleCreateInfo {
                code_size: THEM_SHADERS.len(),
                p_code: THEM_SHADERS.as_ptr() as *const _,
                ..Default::default()
            },
            None,
        )
        .expect("derp");
    dbg!(&shader);
    let dset_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&[
            vk::DescriptorSetLayoutBinding::builder()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .binding(0)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .binding(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ])
        .build();
    let d_set_layout = dev
        .create_descriptor_set_layout(&dset_create_info, None)
        .expect("derp");
    let p_create_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&[d_set_layout])
        .push_constant_ranges(&[vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)
            .build()])
        .build();
    let p_layout = dev
        .create_pipeline_layout(&p_create_info, None)
        .expect("derp");
    dbg!(&p_layout);
    let p_cache = dev
        .create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder().build(), None)
        .expect("derp");
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
    let buf = dev
        .create_buffer(
            &vk::BufferCreateInfo::builder()
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                )
                .size(ROWS * COLS * 4),
            None,
        )
        .expect("derp");
    dbg!(&buf);
    let stage_buf = dev
        .create_buffer(
            &vk::BufferCreateInfo::builder()
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                .size(ROWS * COLS * 4),
            None,
        )
        .expect("derp");
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
    let buf_mem = dev
        .allocate_memory(
            &vk::MemoryAllocateInfo::builder()
                .memory_type_index(find_memory_type(&buf_reqs, &mem_props, default_flags))
                .allocation_size(buf_reqs.size),
            None,
        )
        .expect("derp");
    let stage_mem = dev
        .allocate_memory(
            &vk::MemoryAllocateInfo::builder()
                .memory_type_index(find_memory_type(&stage_reqs, &mem_props, stage_flags))
                .allocation_size(stage_reqs.size),
            None,
        )
        .expect("derp");
    dbg!(&buf_mem);
    dbg!(&stage_mem);
    dev.bind_buffer_memory(buf, buf_mem, 0).expect("derp");
    dev.bind_buffer_memory(stage_buf, stage_mem, 0)
        .expect("derp");
    let ptr = dev
        .map_memory(stage_mem, 0, stage_reqs.size, Default::default())
        .expect("derp");
    {
        let in_float: &mut [f32] =
            std::slice::from_raw_parts_mut(ptr as *mut _, (stage_reqs.size / 4) as usize);
        for y in 0..ROWS {
            for x in 0..COLS {
                in_float[(y * COLS + x) as usize] = (y + x) as f32;
            }
        }
    }
    let descriptor_pool = dev
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&[vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(2)
                    .build()]),
            None,
        )
        .expect("derp");
    let descriptor_sets = dev
        .allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[d_set_layout]),
        )
        .expect("derp");
    dev.update_descriptor_sets(
        &[
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk::DescriptorBufferInfo::builder()
                    .buffer(buf)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[vk::DescriptorBufferInfo::builder()
                    .buffer(buf)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()])
                .build(),
        ],
        &[],
    );

    // oh my god fucking do the thing already
    let cmd = bufs[0];
    dev.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo {
            ..Default::default()
        },
    )
    .expect("derp");
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
    dev.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        p_layout,
        0,
        &[descriptor_sets[0]],
        &[],
    );
    let args = WidthHeight {
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
            std::mem::size_of::<WidthHeight>(),
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
    dev.end_command_buffer(cmd).expect("derp");
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
    // the aristocrats.

    // unsafe fn dadada(out: *mut f32, lhs: *const half, rhs: *const half, m: u32, n: u32, k: u32, ) {
    // dadada(1024, 256, 0, stream)(out, lhs, rhs, m, n, k);
    // foo⟪1024, 256, 0, stream⟫(0);
}

fn main() {
    unsafe { main_unsafe() }
}
