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
    cell::{RefCell, UnsafeCell},
    ffi::c_void,
    mem::MaybeUninit,
    ptr::{self, NonNull},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use std::f64::consts::PI;

use alloc::{collections::BTreeMap, rc::Rc, sync::Arc};
use allocator_api2::{alloc::Allocator, boxed::Box};
use anyhow::bail;
use libspa_sys::{
    spa_audio_info_raw, spa_format_audio_raw_build, spa_pod_builder, spa_pod_builder_init,
    SPA_PARAM_EnumFormat, SPA_AUDIO_FORMAT_F32,
};
use pipewire::{
    properties,
    spa::{spa_interface_call_method, Direction},
    stream::{ListenerBuilderT, Stream, StreamFlags},
    Context, IsSource, MainLoop,
};

use crate::util::{AtomicRing, AtomicRingReader, AtomicRingWaiter, AtomicRingWriter};

const SAMPLE_RATE: u32 = 48000;
const NUM_CHANNELS: usize = 2;
const VOLUME: f32 = 0.1;
const CAPTURE_BUFFER_POOL_SIZE: usize = 10;

struct RealTimeThreadState {
    accumulator: f64,
    did_thing: bool,
    did_other_thing: bool,
    capture_sinks: *mut RtCaptureSink,
}

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
struct RtObjectHeap {
    inner: Rc<RefCell<RtObjectHeapInner>>,
}

impl RtObjectHeap {
    fn new(arena_size: u32, max_arenas: usize) -> Self {
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
unsafe impl Allocator for RtObjectHeap {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        unsafe {
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
            let ptr = NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(
                data_ptr as *mut u8,
                layout.size(),
            ));
            Ok(ptr)
        }
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
                let first_above = arena.free_map.range(addr as u32..).next().map(|(k, v)| (*k, *v));
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

// TODO I feel like I'm definitely dupe'ing SPA/pipewire infra here
// lets revisit this and see if there's a more congruent way to do everything. but xplat
// also will def be a thing so...
struct RtBuffer {
    data: *mut c_void,
    size: usize,
}

unsafe impl Send for RtBuffer {}

impl RtBuffer {
    //TODO
}

struct RtCaptureSink {
    next: *mut RtCaptureSink,
    free_buffers: AtomicRingReader<RtBuffer>,
}

struct ControlThreadState {
    request_rx: AtomicRingReader<AudioControlThreadRequest>,
    response_tx: AtomicRingWriter<AudioControlThreadResponse>,
    response_waiter: AtomicRingWaiter<AudioControlThreadResponse>,
    rt_heap: RtObjectHeap,
}

struct AudioControlThreadInner {
    // Must only be accessed in process callbacks or otherwise on the pw_data_loop.
    // I feel the need to defend this code a bit. Unfortunately, the pipewire rust
    // bindings contain a number of soundness bugs, one of which is that the `process`
    // callback does not require the closure to be Send, even though the callback
    // can run on another thread (the RT thread). I could use RefCell here and just
    // pinky swear that I won't actually access the RefCell on the wrong thread,
    // but given that it's unsafe, I wanted to make it unsafe to get the reference.
    //
    // So we're using the normally internal-to-cell-implementations UnsafeCell here.
    rt: UnsafeCell<RealTimeThreadState>,
    ct: UnsafeCell<ControlThreadState>,
}

enum AudioControlThreadRequest {
    AddCaptureSink {
        interval: Duration,
        waiter: AtomicRingWaiter<AudioPacket>,
        writer: AtomicRingWriter<AudioPacket>,
    },
}

enum AudioControlThreadResponse {
    Initialized {
        control_event: *mut libspa_sys::spa_source,
        quit_event: *mut libspa_sys::spa_source,
        pw_loop: *mut pipewire_sys::pw_loop,
    },
    InitializationFailed, // TODO wrap an error
    CaptureStream {
        free_buffers: AtomicRingWriter<RtBuffer>,
        waiter: AtomicRingWaiter<RtBuffer>,
    },
}

unsafe impl Send for AudioControlThreadResponse {}

struct AudioPacket;

struct AudioStream;

struct AudioControlThread {
    thread: Option<std::thread::JoinHandle<()>>,
    waiter: AtomicRingWaiter<AudioControlThreadResponse>,
    request_tx: AtomicRingWriter<AudioControlThreadRequest>,
    control_event: *mut libspa_sys::spa_source,
    quit_event: *mut libspa_sys::spa_source,
    pw_loop: *mut pipewire_sys::pw_loop,
}

fn sine_wave(last: &mut f64, buf: &mut [f32], n_channels: usize) {
    for i in 0..buf.len() / n_channels {
        *last += PI * 2.0 * 440.0 / SAMPLE_RATE as f64;
        if *last > PI * 2.0 {
            *last -= PI * 2.0;
        }
        let val = (*last).sin() as f32 * VOLUME;
        for j in 0..n_channels {
            buf[i * n_channels + j] = val;
        }
    }
}

unsafe fn audio_control_thread_main(state: Rc<AudioControlThreadInner>) -> anyhow::Result<()> {
    let main_loop = MainLoop::new()?;
    let context = Context::new(&main_loop)?;
    let core = context.connect(None)?;
    // we do not use the provided userdata mechanism because
    // 1. it is jank af with respect to Default trait bounds
    // 2. it's actually unsound because process is running on a separate thread
    let mut capture_stream: Stream<()> = Stream::new(
        &core,
        "audio-capture",
        properties! {
            *pipewire::keys::MEDIA_TYPE => "Audio",
            *pipewire::keys::MEDIA_CATEGORY => "Capture",
            *pipewire::keys::MEDIA_ROLE => "Communication",
            *pipewire::keys::NODE_LATENCY => "480/48000",
        },
    )?;
    let mut render_stream: Stream<()> = Stream::new(
        &core,
        "audio-render",
        properties! {
            *pipewire::keys::MEDIA_TYPE => "Audio",
            *pipewire::keys::MEDIA_CATEGORY => "Playback",
            *pipewire::keys::MEDIA_ROLE => "Communication",
            *pipewire::keys::NODE_LATENCY => "480/48000",
        },
    )?;
    let mut builder: spa_pod_builder = std::mem::zeroed();
    let mut buf = [0u8; 1024];
    spa_pod_builder_init(&mut builder, buf.as_mut_ptr() as *mut c_void, buf.len() as u32);
    let pod = spa_format_audio_raw_build(
        &mut builder,
        SPA_PARAM_EnumFormat,
        &mut spa_audio_info_raw {
            format: SPA_AUDIO_FORMAT_F32,
            rate: SAMPLE_RATE,
            channels: NUM_CHANNELS as u32,
            ..std::mem::zeroed()
        },
    );
    let _capture_listener = {
        capture_stream
            .add_local_listener()
            .process({
                let state = state.clone();
                move |stream, ()| {
                    // much yo, very lo
                    let rt = &mut *state.rt.get();
                    rt.did_thing = true;
                    if let Some(mut buffer) = stream.dequeue_buffer() {
                        for data in buffer.datas_mut() {
                            //
                        }
                    } else {
                        println!("null buffer omg what do we do");
                    }
                }
            })
            .param_changed(|id, (), pod| {
                println!("capture params {:?} {} {:?}", std::thread::current(), id, pod);
            })
            .state_changed(|old, new| {
                println!("capture state changed {:?} {:?} {:?}", std::thread::current(), old, new);
            })
            .register()?
    };
    let _render_listener = render_stream
        .add_local_listener()
        .process({
            let state = state.clone();
            move |stream, ()| {
                let rt = &mut *state.rt.get();
                // TODO we're using raw buffers here due to some limitations in
                // the safe rust bindings, namely, the lack of access to
                // 'requested'
                let raw_buffer = stream.dequeue_raw_buffer();
                if raw_buffer.is_null() {
                    return;
                }
                let requested = (*raw_buffer).requested as usize;
                assert!((*(*raw_buffer).buffer).n_datas > 0);
                let data0 = &mut *(*(*raw_buffer).buffer).datas;
                if data0.data.is_null() {
                    return;
                }
                let stride = NUM_CHANNELS * std::mem::size_of::<f32>();
                let data_buf = std::slice::from_raw_parts_mut(
                    data0.data as *mut f32,
                    (data0.maxsize as usize / std::mem::size_of::<f32>())
                        .min(requested * NUM_CHANNELS),
                );
                sine_wave(&mut rt.accumulator, data_buf, NUM_CHANNELS);
                (*data0.chunk).offset = 0;
                (*data0.chunk).stride = stride as i32;
                (*data0.chunk).size = (data_buf.len() * std::mem::size_of::<f32>()) as u32;
                if rt.did_thing && !rt.did_other_thing {
                    rt.did_other_thing = true;
                    println!("did other thing after doing thing");
                }
                stream.queue_raw_buffer(raw_buffer);
            }
        })
        .param_changed(|id, _, pod| {
            println!("render params {:?} {} {:?}", std::thread::current(), id, pod);
        })
        .state_changed(|old, new| {
            println!("render state changed {:?} {:?} {:?}", std::thread::current(), old, new);
        })
        .register()?;
    capture_stream.connect(
        Direction::Input,
        None,
        StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
        &mut [pod],
    )?;
    render_stream.connect(
        Direction::Output,
        None,
        StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
        &mut [pod],
    )?;
    let control_event = main_loop.add_event({
        let state = state.clone();
        move || {
            let ct = &mut *state.ct.get();
            while let Some(request) = ct.request_rx.try_pop() {
                match request {
                    AudioControlThreadRequest::AddCaptureSink { interval, waiter, writer } => {
                        let rate = (SAMPLE_RATE as f32 * interval.as_secs_f32()) as usize;
                        let buf_len = rate * NUM_CHANNELS * std::mem::size_of::<f32>();

                        for _i in 0..CAPTURE_BUFFER_POOL_SIZE {
                            let buf: Box<[MaybeUninit<f32>], RtObjectHeap> =
                                Box::new_zeroed_slice_in(buf_len, ct.rt_heap.clone());
                            let buf =
                                Box::<[f32], RtObjectHeap>::into_raw(unsafe { buf.assume_init() });

                            todo!();
                        }
                    }
                }
            }
            println!("event handler");
        }
    });
    // Another API issue: add_event callbacks aren't FnMut
    let received_quit = Arc::new(AtomicBool::new(false));
    let quit_event = main_loop.add_event({
        let main_loop = main_loop.clone();
        let received_quit = received_quit.clone();
        move || {
            println!("quit event!");
            received_quit.store(true, Ordering::SeqCst);
            main_loop.quit();
        }
    });

    assert!((*state.ct.get())
        .response_tx
        .try_push(AudioControlThreadResponse::Initialized {
            control_event: control_event.as_ptr(),
            quit_event: quit_event.as_ptr(),
            pw_loop: main_loop.as_raw() as *const _ as *mut _
        })
        .is_none());
    (*state.ct.get()).response_waiter.alert();
    main_loop.run();

    if !received_quit.load(Ordering::SeqCst) {
        panic!("pipewire main loop exited unexpectedly");
    }
    Ok(())
}

unsafe fn signal_event(pw_loop: *mut pipewire_sys::pw_loop, event: *mut libspa_sys::spa_source) {
    use libspa_sys as spa_sys;
    spa_interface_call_method!(
        &mut (*(*pw_loop).utils).iface as *mut libspa_sys::spa_interface,
        spa_sys::spa_loop_utils_methods,
        signal_event,
        event
    );
}

impl AudioControlThread {
    pub fn new() -> anyhow::Result<Self> {
        // single-slot mailbox for receiving responses back from the ACT
        // prob not the ideal solution here but I had it handy and I'm sick of
        // adding dependencies for this ... thing. will likely switch over to using
        // thread parking so we can have multiple in flight requests.
        let (response_rx, response_tx) = AtomicRing::new(1);
        let (request_rx, request_tx) = AtomicRing::new(1);
        let waiter = AtomicRingWaiter::new(response_rx);
        let thread = Some(std::thread::spawn({
            let waiter = waiter.clone();
            move || unsafe {
                let state = Rc::new(AudioControlThreadInner {
                    rt: UnsafeCell::new(RealTimeThreadState {
                        accumulator: 0.0,
                        did_thing: false,
                        did_other_thing: false,
                        capture_sinks: ptr::null_mut(),
                    }),
                    ct: UnsafeCell::new(ControlThreadState {
                        request_rx,
                        response_tx,
                        response_waiter: waiter,
                        rt_heap: RtObjectHeap::new(1024 * 1024, 8),
                    }),
                });
                if let Err(_e) = audio_control_thread_main(state.clone()) {
                    assert!((*state.ct.get())
                        .response_tx
                        .try_push(AudioControlThreadResponse::InitializationFailed)
                        .is_none());
                    (*state.ct.get()).response_waiter.alert();
                }
            }
        }));
        match waiter.wait_pop() {
            AudioControlThreadResponse::Initialized { control_event, quit_event, pw_loop } => {
                Ok(Self { thread, waiter, control_event, quit_event, pw_loop, request_tx })
            }
            AudioControlThreadResponse::InitializationFailed => {
                bail!("failed to initialize audio control thread")
            }
            _ => unreachable!("invalid response"),
        }
    }

    fn try_send_request(&mut self, request: AudioControlThreadRequest) -> anyhow::Result<()> {
        if self.request_tx.try_push(request).is_some() {
            bail!("failed to send request to audio control thread");
        }
        unsafe { signal_event(self.pw_loop, self.control_event) }
        Ok(())
    }

    pub fn capture_audio_stream(&mut self, interval: Duration) -> anyhow::Result<AudioStream> {
        let (response_rx, response_tx) = AtomicRing::new(64);
        let response_waiter = AtomicRingWaiter::new(response_rx);
        let request = AudioControlThreadRequest::AddCaptureSink {
            interval,
            waiter: response_waiter,
            writer: response_tx,
        };
        self.try_send_request(request)?;
        let response = self.waiter.wait_pop();
        todo!();
    }
}

impl Drop for AudioControlThread {
    fn drop(&mut self) {
        unsafe { signal_event(self.pw_loop, self.quit_event) }
        self.thread.take().unwrap().join().unwrap();
    }
}

pub fn test() -> anyhow::Result<()> {
    for _ in 0..100 {
        std::thread::spawn(|| {
            let _mainloop = pipewire::MainLoop::new().unwrap();
        });
    }

    let mut audio_control = AudioControlThread::new()?;
    println!("we got audio control");
    let audio_stream = audio_control.capture_audio_stream(Duration::from_millis(10))?;
    println!("we got capture stream");
    std::thread::sleep(Duration::from_secs(5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use allocator_api2::vec::*;

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
        let rt_alloc = RtObjectHeap::new(1024 * 1024, 8);
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
