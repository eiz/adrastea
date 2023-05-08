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

unsafe fn find_compute_queue_family(instance: &ash::Instance, phys_dev: vk::PhysicalDevice) -> u32 {
    let props = instance.get_physical_device_queue_family_properties(phys_dev);

    for (i, fam) in props.iter().enumerate() {
        if fam.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            return i as u32;
        }
    }

    panic!("oh noes couldn't find a compute queue");
}

unsafe fn main_unsafe() {
    println!("The endless sea.");
    let entry = Entry::load().expect("failed to load vulkan");
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 3, 0),
        ..Default::default()
    };
    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        ..Default::default()
    };
    let instance = entry.create_instance(&create_info, None).expect("derp");
    dbg!(instance.handle());
    let phys_devs = instance.enumerate_physical_devices().expect("derp");
    dbg!(&phys_devs);
    let queue_family_index = find_compute_queue_family(&instance, phys_devs[0]);
    let queue_infos = [vk::DeviceQueueCreateInfo {
        queue_count: 1,
        queue_family_index,
        ..Default::default()
    }];
    let create_info = vk::DeviceCreateInfo {
        queue_create_info_count: 1,
        p_queue_create_infos: queue_infos.as_ptr(),
        ..Default::default()
    };
    let dev = instance
        .create_device(phys_devs[0], &create_info, None)
        .expect("derp");
    dbg!(dev.handle());
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
    let buf = bufs[0];
    dev.begin_command_buffer(
        buf,
        &vk::CommandBufferBeginInfo {
            ..Default::default()
        },
    )
    .expect("derp");
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
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .build(),
        ])
        .build();
    let d_set_layout = dev
        .create_descriptor_set_layout(&dset_create_info, None)
        .expect("derp");
    let p_create_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&[d_set_layout])
        .push_constant_ranges(&[vk::PushConstantRange::builder().offset(0).size(8).build()])
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
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                .size(1024 * 1024 * 4)
                .build(),
            None,
        )
        .expect("derp");
    dbg!(&buf);
    let stage_buf = dev
        .create_buffer(
            &vk::BufferCreateInfo::builder()
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .size(1024 * 1024 * 4)
                .build(),
            None,
        )
        .expect("derp");
    dbg!(&stage_buf);
    let mem_props = instance.get_physical_device_memory_properties(phys_devs[0]);
    dbg!(&mem_props);

    // unsafe fn dadada(out: *mut f32, lhs: *const half, rhs: *const half, m: u32, n: u32, k: u32, ) {
    // dadada(1024, 256, 0, stream)(out, lhs, rhs, m, n, k);
    // foo⟪1024, 256, 0, stream⟫(0);
}

fn main() {
    unsafe { main_unsafe() }
}
