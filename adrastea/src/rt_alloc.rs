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
    alloc::Layout,
    cell::RefCell,
    ffi::c_void,
    ptr::{self, NonNull},
};

use alloc::{collections::BTreeMap, rc::Rc};
use allocator_api2::alloc::Allocator;

struct RtObjectArena {
    arena: *mut c_void,
    size: u32,
    free_map: BTreeMap<u32, u32>,
}

impl RtObjectArena {
    pub fn new(size: u32) -> Self {
        unsafe {
            let ptr = libc::mmap64(
                ptr::null_mut(),
                size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if ptr == libc::MAP_FAILED {
                panic!("rt_allocator: failed to alloc pages");
            }
            if libc::mlock(ptr, size as usize) < 0 {
                panic!("rt_allocator: failed to lock pages");
            }
            let mut free_map = BTreeMap::new();
            free_map.insert(0, size);
            Self { arena: ptr, size, free_map }
        }
    }

    pub fn split_and_allocate(&mut self, offset: u32, layout: Layout) -> *mut c_void {
        let size = self.free_map.remove(&offset).expect("invariant: invalid allocation offset");
        let size_lo = required_padding(offset, layout.align() as u32);
        let data_base = offset + size_lo;
        let upper_base = data_base + layout.size() as u32;
        let size_hi = offset + size - upper_base;
        assert!(upper_base + size_hi <= self.size);
        unsafe {
            let result = self.arena.add(data_base as usize);
            if size_lo > 0 {
                self.free_map.insert(offset, size_lo);
            }
            if size_hi > 0 {
                self.free_map.insert(upper_base, size_hi);
            }
            result
        }
    }
}

impl Drop for RtObjectArena {
    fn drop(&mut self) {
        unsafe {
            libc::munlock(self.arena, self.size as usize);
            libc::munmap(self.arena, self.size as usize);
        }
    }
}

struct RtObjectHeapInner {
    arenas: Vec<RtObjectArena>,
    arena_size: u32,
    max_arenas: usize,
}

impl RtObjectHeapInner {
    fn more_core(&mut self) -> usize {
        if self.arenas.len() >= self.max_arenas {
            panic!("rt_allocator: exhausted arena limit");
        }
        self.arenas.push(RtObjectArena::new(self.arena_size));
        self.arenas.len() - 1
    }
}

#[derive(Clone)]
pub struct RtObjectHeap {
    inner: Rc<RefCell<RtObjectHeapInner>>,
}

impl RtObjectHeap {
    pub fn new(arena_size: u32, max_arenas: usize) -> Self {
        Self {
            inner: Rc::new(RefCell::new(RtObjectHeapInner {
                arenas: Vec::new(),
                arena_size,
                max_arenas,
            })),
        }
    }
}

fn required_padding(offset: u32, align: u32) -> u32 {
    if offset % align == 0 {
        0
    } else {
        align - (offset % align)
    }
}

// so i started writing like YE OLDE BEST FIT ALLOCATOR WITH LINKED LISTS
// and that was annoying
// so i thought to myself
// "BTreeMap"
// 🤣
// anyway its super jank nonsense
// allocations and deallocations obviously use the global heap so they are not realtime
// but realtime is expected to do no allocations. the use pattern here is to allocate
// stuff on the control thread and then pass raw pointers to RT
unsafe impl Allocator for RtObjectHeap {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let mut inner = self.inner.borrow_mut();
        if layout.size() > u32::MAX as usize || layout.size() > inner.arena_size as usize {
            panic!("rt_allocator: alloc size {} too large", layout.size());
        }
        let mut best = None;
        for (i, arena) in &mut inner.arenas.iter().enumerate() {
            for (&offset, &size) in &arena.free_map {
                let padding = required_padding(offset, layout.align() as u32);
                let total_size = layout.size() as u32 + padding;
                if total_size > size {
                    continue;
                }
                match best {
                    None => {
                        best = Some((i, offset, size));
                    }
                    Some((_, _, best_size)) if size < best_size => {
                        best = Some((i, offset, size));
                    }
                    _ => {}
                }
            }
        }
        let data_ptr = if let Some((arena_idx, offset, _)) = best {
            inner.arenas[arena_idx].split_and_allocate(offset, layout)
        } else {
            let idx = inner.more_core();
            inner.arenas[idx].split_and_allocate(0, layout)
        };
        let ptr = unsafe {
            NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(
                data_ptr as *mut u8,
                layout.size(),
            ))
        };
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let mut inner = self.inner.borrow_mut();
        let addr = ptr.as_ptr() as usize;

        for arena in &mut inner.arenas {
            let arena_base = arena.arena as usize;
            let arena_end = arena_base + arena.size as usize;
            if addr >= arena_base && addr < arena_end {
                let addr = addr - arena_base;
                let mut free_start = addr as u32;
                let mut free_end = free_start + layout.size() as u32;
                let first_below =
                    arena.free_map.range(..addr as u32).next_back().map(|(k, v)| (*k, *v));
                let first_above =
                    arena.free_map.range(addr as u32 + 1..).next().map(|(k, v)| (*k, *v));
                if let Some((below_offset, below_size)) = first_below {
                    let below_end = below_offset + below_size;
                    if below_end == addr as u32 {
                        free_start = below_offset;
                    }
                    arena.free_map.remove(&below_offset);
                }
                if let Some((above_offset, above_size)) = first_above {
                    if above_offset == free_end {
                        free_end = above_offset + above_size;
                    }
                    arena.free_map.remove(&above_offset);
                }
                arena.free_map.insert(free_start, free_end - free_start);
            }
        }
    }
}
