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

use ash::{vk, Entry};

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
    let queue_infos = [vk::DeviceQueueCreateInfo {
        queue_count: 1,
        queue_family_index: find_compute_queue_family(&instance, phys_devs[0]),
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
}

fn main() {
    unsafe { main_unsafe() }
}
