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
    any::{request_ref, Provider},
    fmt::Debug,
    marker::PhantomData,
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};

use alloc::{alloc::dealloc, sync::Arc};
use parking_lot::{Condvar, Mutex};

pub trait IUnknown: Provider + Debug {}

impl dyn IUnknown + '_ {
    pub fn query_interface<T: ?Sized + 'static>(&self) -> Option<&T> {
        request_ref(self)
    }
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

pub fn ceil_div(a: u64, b: u64) -> u64 {
    (a + b - 1) / b
}

pub struct AtomicRing<T: Send> {
    buf: *mut MaybeUninit<T>,
    length: usize,
    read_ptr: AtomicUsize,
    write_ptr: AtomicUsize,
    _dead: PhantomData<T>,
}

unsafe impl<T: Send> Send for AtomicRing<T> {}
unsafe impl<T: Send> Sync for AtomicRing<T> {}

impl<T: Send> AtomicRing<T> {
    pub fn new(length: usize) -> (AtomicRingReader<T>, AtomicRingWriter<T>) {
        assert!(length.is_power_of_two());

        let state = Arc::new(Self {
            buf: unsafe { alloc::alloc::alloc(Layout::array::<MaybeUninit<T>>(length).unwrap()) }
                as *mut MaybeUninit<T>,
            length: length,
            read_ptr: AtomicUsize::new(0),
            write_ptr: AtomicUsize::new(0),
            _dead: PhantomData,
        });

        (AtomicRingReader(state.clone()), AtomicRingWriter(state))
    }
}

impl<T: Send> Drop for AtomicRing<T> {
    fn drop(&mut self) {
        let mut read_ptr = self.read_ptr.load(Ordering::SeqCst);
        let write_ptr = self.write_ptr.load(Ordering::SeqCst);

        while read_ptr < write_ptr {
            let read_masked = read_ptr & (self.length - 1);

            unsafe {
                drop((*self.buf.offset(read_masked as isize)).assume_init_read());
            }

            read_ptr += 1;
        }

        unsafe {
            dealloc(self.buf as *mut u8, Layout::array::<MaybeUninit<T>>(self.length).unwrap())
        }
    }
}

pub struct AtomicRingReader<T: Send>(Arc<AtomicRing<T>>);
pub struct AtomicRingWriter<T: Send>(Arc<AtomicRing<T>>);

impl<T: Send> AtomicRingReader<T> {
    pub fn read_available(&self) -> usize {
        let read_ptr = self.0.read_ptr.load(Ordering::Acquire);
        let write_ptr = self.0.write_ptr.load(Ordering::Relaxed);

        write_ptr - read_ptr
    }

    pub fn try_pop(&mut self) -> Option<T> {
        let read_ptr = self.0.read_ptr.load(Ordering::Acquire);
        let write_ptr = self.0.write_ptr.load(Ordering::Relaxed);
        let read_masked = read_ptr & (self.0.length - 1);

        if write_ptr == read_ptr {
            return None;
        }

        let result = unsafe { (*self.0.buf.offset(read_masked as isize)).assume_init_read() };
        self.0.read_ptr.store(read_ptr.wrapping_add(1), Ordering::Release);
        Some(result)
    }
}

impl<T: Send> AtomicRingWriter<T> {
    pub fn write_available(&self) -> usize {
        let read_ptr = self.0.read_ptr.load(Ordering::Relaxed);
        let write_ptr = self.0.write_ptr.load(Ordering::Acquire);

        self.0.length - (write_ptr - read_ptr)
    }
    pub fn try_push(&mut self, value: T) -> Option<T> {
        let read_ptr = self.0.read_ptr.load(Ordering::Relaxed);
        let write_ptr = self.0.write_ptr.load(Ordering::Acquire);
        let write_masked = write_ptr & (self.0.length - 1);

        if write_ptr - read_ptr == self.0.length {
            return Some(value);
        }

        unsafe { (*self.0.buf.offset(write_masked as isize)).write(value) };
        self.0.write_ptr.store(write_ptr.wrapping_add(1), Ordering::Release);
        None
    }
}

impl<T: Send + Clone> AtomicRingWriter<T> {
    pub fn try_pushn(&mut self, value: &[T]) -> usize {
        let mut copied = 0;
        let read_ptr = self.0.read_ptr.load(Ordering::Relaxed);
        let write_ptr = self.0.write_ptr.load(Ordering::Acquire);
        let to_write = self.0.length - (write_ptr - read_ptr);

        for item in value {
            if copied == to_write {
                break;
            }

            let write_masked = (write_ptr + copied) & (self.0.length - 1);

            unsafe { (*self.0.buf.offset(write_masked as isize)).write(item.clone()) };
            copied += 1;
        }

        self.0.write_ptr.store(write_ptr.wrapping_add(copied), Ordering::Release);
        copied
    }
}

pub struct AtomicRingWaiterInner<T: Send> {
    ring: Mutex<AtomicRingReader<T>>,
    cvar: Condvar,
}

pub struct AtomicRingWaiter<T: Send>(Arc<AtomicRingWaiterInner<T>>);

impl<T: Send> AtomicRingWaiter<T> {
    pub fn new(ring: AtomicRingReader<T>) -> Self {
        Self(Arc::new(AtomicRingWaiterInner { ring: Mutex::new(ring), cvar: Condvar::new() }))
    }

    pub fn read_available(&self) -> usize {
        self.0.ring.lock().read_available()
    }

    pub fn try_pop(&self) -> Option<T> {
        self.0.ring.lock().try_pop()
    }

    pub fn wait_pop(&self) -> T {
        let mut ring = self.0.ring.lock();

        loop {
            if let Some(value) = ring.try_pop() {
                return value;
            }

            self.0.cvar.wait(&mut ring);
        }
    }

    pub fn alert(&self) {
        self.0.cvar.notify_one();
    }
}

impl<T: Send> Clone for AtomicRingWaiter<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
