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

// so i started writing like YE OLDE BEST FIT ALLOCATOR WITH LINKED LISTS
// and that was annoying
// so i thought to myself
// "BTreeMap"
// 🤣
// anyway its super jank nonsense
// allocations and deallocations obviously use the global heap for the free list
// so they are not realtime but realtime is expected to do no allocations. the use
// pattern here is to allocate stuff on the control thread and then pass raw pointers
// to RT
//
// this can also be used as a manager for other 'weird' memory resources like
// memfds, wayland buffers, sub allocations of gpu buffers, etc

use core::{
    alloc::Layout,
    cell::RefCell,
    ffi::c_void,
    fmt::{self, Formatter},
    ptr::{self, NonNull},
};

use alloc::{collections::BTreeMap, rc::Rc};
use allocator_api2::alloc::Allocator;

fn required_padding(offset: usize, align: usize) -> usize {
    if offset % align == 0 {
        0
    } else {
        align - (offset % align)
    }
}

pub trait ArenaHandle {
    fn as_ptr(&self) -> *mut c_void;
    unsafe fn offset(&self, offset: usize) -> Self;
}

impl ArenaHandle for *mut c_void {
    fn as_ptr(&self) -> *mut c_void {
        *self
    }

    unsafe fn offset(&self, offset: usize) -> Self {
        unsafe { self.add(offset) }
    }
}

pub trait ArenaAllocator {
    type Handle: ArenaHandle;
    type Params: Clone;

    fn allocate(size: usize, params: &Self::Params) -> Self::Handle;
    unsafe fn deallocate(handle: &Self::Handle, size: usize, params: &Self::Params);
}

#[derive(Copy, Clone, Debug)]
pub struct LockedArenaAllocator;

impl ArenaAllocator for LockedArenaAllocator {
    type Handle = *mut c_void;
    type Params = ();

    fn allocate(size: usize, _params: &Self::Params) -> Self::Handle {
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
            ptr
        }
    }

    unsafe fn deallocate(handle: &Self::Handle, size: usize, _params: &Self::Params) {
        unsafe {
            libc::munlock(handle.as_ptr(), size as usize);
            libc::munmap(handle.as_ptr(), size as usize);
        }
    }
}

struct RtObjectArena<T: ArenaAllocator = LockedArenaAllocator> {
    arena: T::Handle,
    arena_params: T::Params,
    size: usize,
    free_map: BTreeMap<usize, usize>,
}

impl<T: ArenaAllocator> RtObjectArena<T> {
    pub fn new(size: usize, params: &T::Params) -> Self {
        let mut free_map = BTreeMap::new();
        free_map.insert(0, size);
        Self { arena: T::allocate(size, params), arena_params: params.clone(), size, free_map }
    }

    pub fn split_and_allocate(&mut self, offset: usize, layout: Layout) -> T::Handle {
        let size = self.free_map.remove(&offset).expect("invariant: invalid allocation offset");
        let size_lo = required_padding(offset, layout.align() as usize);
        let data_base = offset + size_lo;
        let upper_base = data_base + layout.size() as usize;
        let size_hi = offset + size - upper_base;
        assert!(upper_base + size_hi <= self.size);
        unsafe {
            let result = self.arena.offset(data_base as usize);
            if size_lo > 0 {
                self.free_map.insert(offset, size_lo);
            }
            if size_hi > 0 {
                self.free_map.insert(upper_base, size_hi);
            }
            result
        }
    }

    pub fn try_deallocate_and_merge(&mut self, ptr: *mut c_void, layout: Layout) -> bool {
        let addr = ptr as usize;
        let arena_base = self.arena.as_ptr() as usize;
        let arena_end = arena_base + self.size as usize;
        if addr >= arena_base && addr < arena_end {
            let addr = addr - arena_base;
            let mut free_start = addr as usize;
            let mut free_end = free_start + layout.size() as usize;
            let first_below =
                self.free_map.range(..addr as usize).next_back().map(|(k, v)| (*k, *v));
            let first_above =
                self.free_map.range(addr as usize + 1..).next().map(|(k, v)| (*k, *v));
            if let Some((below_offset, below_size)) = first_below {
                let below_end = below_offset + below_size;
                if below_end == addr as usize {
                    free_start = below_offset;
                }
                self.free_map.remove(&below_offset);
            }
            if let Some((above_offset, above_size)) = first_above {
                if above_offset == free_end {
                    free_end = above_offset + above_size;
                }
                self.free_map.remove(&above_offset);
            }
            self.free_map.insert(free_start, free_end - free_start);
            return true;
        }
        false
    }
}

impl<T: ArenaAllocator> Drop for RtObjectArena<T> {
    fn drop(&mut self) {
        unsafe { T::deallocate(&self.arena, self.size, &self.arena_params) }
    }
}

impl<T: ArenaAllocator> fmt::Debug for RtObjectArena<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RtObjectArena").field("size", &self.size).finish()
    }
}

#[derive(Debug)]
struct RtObjectHeapInner<T: ArenaAllocator> {
    arenas: Vec<RtObjectArena<T>>,
    arena_params: T::Params,
    arena_size: usize,
    max_arenas: usize,
}

impl<T: ArenaAllocator> RtObjectHeapInner<T> {
    fn more_core(&mut self) -> usize {
        if self.arenas.len() >= self.max_arenas {
            panic!("rt_allocator: exhausted arena limit");
        }
        self.arenas.push(RtObjectArena::new(self.arena_size, &self.arena_params));
        self.arenas.len() - 1
    }
}

#[derive(Clone)]
pub struct RtObjectHeap<T: ArenaAllocator = LockedArenaAllocator> {
    inner: Rc<RefCell<RtObjectHeapInner<T>>>,
}

impl<T: ArenaAllocator> RtObjectHeap<T> {
    pub fn new(arena_size: usize, max_arenas: usize, arena_params: T::Params) -> Self {
        Self {
            inner: Rc::new(RefCell::new(RtObjectHeapInner {
                arenas: Vec::new(),
                arena_size,
                arena_params,
                max_arenas,
            })),
        }
    }

    pub fn allocate_handle(
        &self, layout: Layout,
    ) -> Result<T::Handle, allocator_api2::alloc::AllocError> {
        let mut inner = self.inner.borrow_mut();
        if layout.size() > usize::MAX as usize || layout.size() > inner.arena_size as usize {
            panic!("rt_allocator: alloc size {} too large", layout.size());
        }
        let mut best = None;
        for (i, arena) in &mut inner.arenas.iter().enumerate() {
            for (&offset, &size) in &arena.free_map {
                let padding = required_padding(offset, layout.align() as usize);
                let total_size = layout.size() as usize + padding;
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
        let handle = if let Some((arena_idx, offset, _)) = best {
            inner.arenas[arena_idx].split_and_allocate(offset, layout)
        } else {
            let idx = inner.more_core();
            inner.arenas[idx].split_and_allocate(0, layout)
        };
        Ok(handle)
    }

    pub unsafe fn deallocate_handle(&self, handle: T::Handle, layout: Layout) {
        let mut inner = self.inner.borrow_mut();
        for arena in &mut inner.arenas {
            if arena.try_deallocate_and_merge(handle.as_ptr() as *mut c_void, layout) {
                break;
            }
        }
    }
}

impl<T: ArenaAllocator> fmt::Debug for RtObjectHeap<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RtObjectHeap").finish()
    }
}

unsafe impl<T: ArenaAllocator> Allocator for RtObjectHeap<T> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let data_ptr = self.allocate_handle(layout)?;
        let ptr = unsafe {
            NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(
                data_ptr.as_ptr() as *mut u8,
                layout.size(),
            ))
        };
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let mut inner = self.inner.borrow_mut();
        for arena in &mut inner.arenas {
            if arena.try_deallocate_and_merge(ptr.as_ptr() as *mut c_void, layout) {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use allocator_api2::{boxed::*, vec::*};

    #[test]
    fn allocator_misc() {
        struct Foo {
            a: u32,
            b: u32,
        }
        impl Drop for Foo {
            fn drop(&mut self) {
                println!("dropping foo {} {}", self.a, self.b);
            }
        }
        let rt_alloc: RtObjectHeap = RtObjectHeap::new(1024 * 1024, 8, ());
        let foo = Box::new_in(Foo { a: 1, b: 2 }, rt_alloc.clone());
        let bar = Box::new_in(Foo { a: 3, b: 4 }, rt_alloc.clone());
        for _t in 0..10 {
            let mut test_vec = Vec::new_in(rt_alloc.clone());

            for i in 0..1000 {
                test_vec.push(i);
            }
            for (i, j) in test_vec.iter().enumerate() {
                assert_eq!(i, *j);
            }
        }

        assert_eq!(foo.a, 1);
        assert_eq!(foo.b, 2);
        assert_eq!(bar.a, 3);
        assert_eq!(bar.b, 4);
    }
}
