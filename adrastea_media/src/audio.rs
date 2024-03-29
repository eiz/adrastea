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
    cell::UnsafeCell,
    ffi::c_void,
    mem::MaybeUninit,
    ptr,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use adrastea_core::{
    rt_alloc::RtObjectHeap,
    util::{AtomicRing, AtomicRingReader, AtomicRingWaiter, AtomicRingWriter},
};
use alloc::{rc::Rc, sync::Arc};
use allocator_api2::boxed::Box;
use anyhow::bail;
use libspa_sys::{
    spa_audio_info_raw, spa_format_audio_raw_build, spa_pod_builder, spa_pod_builder_init,
    SPA_PARAM_EnumFormat, SPA_AUDIO_FORMAT_F32,
};
use parking_lot::Mutex;
use pipewire::{
    properties,
    spa::{spa_interface_call_method, Direction},
    stream::{ListenerBuilderT, Stream, StreamFlags},
    Context, IsSource, MainLoop,
};

// TODO: The current version of pipewire-rs, while a good effort, still has a number
// of soundness and usability problems. This code has a bunch of even uglier workarounds
// for those problems. Think through some good design improvements and contribute them.
//
// Overall, at the moment this file is basically a C program with Rust syntax. Proceed with
// _extreme_ caution.
//
// TODO: In some ways this module really wants to recreate something like
// pipewire inside of itself. This seems kind of needed as long as PW is
// Linux-only, but it seems like a bit of a shame. Windows for example has the
// exact set of primitives you need to make it work (eventfd -> HEVENT, memoryfd
// -> CreateFileMapping/ZwCreateSection, dma-buf -> NTHANDLE-style DXGI shared
// textures) except you need something like the SAR handle queue or Chromium's
// Mojo broker to do the cross process handle shipping. For a purely client side
// implementation tho it would work fine.

pub const SAMPLE_RATE: u32 = 16000;
pub const NUM_CHANNELS: usize = 1;
const CAPTURE_BUFFER_POOL_SIZE: usize = 64;

// TODO: Used to launder raw pointers as Send atm. Should go away eventually in favor of
// a more specific interface for RT data.
struct RtPtr<T>(*mut T);
impl<T> RtPtr<T> {
    pub fn into_raw(self) -> *mut T {
        self.0
    }
}
unsafe impl<T> Send for RtPtr<T> {}
impl<T> Clone for RtPtr<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<T> Copy for RtPtr<T> {}

// TODO: Another really "bad idea" helper type. This erases the allocator a buffer
// was allocated from, so that we can send it to the RT thread. As a result,
// RtBuffer does not implemenent `Drop` and will leak if dropped.
struct RtBuffer(*mut [f32]);

unsafe impl Send for RtBuffer {}

impl RtBuffer {
    pub fn new(size: usize, heap: RtObjectHeap) -> Self {
        let buf: Box<[MaybeUninit<f32>], RtObjectHeap> = Box::new_zeroed_slice_in(size, heap);
        let buf = Box::<[f32], RtObjectHeap>::into_raw(unsafe { buf.assume_init() });
        Self(buf)
    }

    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        unsafe { &mut *self.0 }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { &*self.0 }
    }
}

struct RtCaptureSink {
    next: *mut RtCaptureSink,
    object_id: u64,
    pending_buffer: Option<RtBuffer>,
    pending_buffer_offset: usize,
    free_buffers: AtomicRingReader<RtBuffer>,
    waiter: AtomicRingWaiter<AudioPacket>,
    writer: AtomicRingWriter<AudioPacket>,
}

impl RtCaptureSink {
    pub fn new(
        object_id: u64, free_buffers: AtomicRingReader<RtBuffer>,
        waiter: AtomicRingWaiter<AudioPacket>, writer: AtomicRingWriter<AudioPacket>,
    ) -> Self {
        Self {
            next: ptr::null_mut(),
            pending_buffer: None,
            pending_buffer_offset: 0,
            object_id,
            free_buffers,
            waiter,
            writer,
        }
    }
}

struct RealTimeThreadState {
    capture_sinks: *mut RtCaptureSink,
}

impl RealTimeThreadState {
    pub fn new() -> Self {
        Self { capture_sinks: ptr::null_mut() }
    }

    pub unsafe fn process_capture_stream(&mut self, data: &mut [f32]) {
        let mut capture_sink = self.capture_sinks;
        while !capture_sink.is_null() {
            let mut data = &mut *data;
            while !data.is_empty() {
                if (*capture_sink).pending_buffer.is_none() {
                    (*capture_sink).pending_buffer = (*capture_sink).free_buffers.try_pop();
                    if (*capture_sink).pending_buffer.is_none() {
                        println!("xrun: exhausted capture buffers");
                        break;
                    }
                }
                let pending_buffer = (*capture_sink).pending_buffer.as_mut().unwrap();
                let pending_buffer = pending_buffer.as_slice_mut();
                let avail = pending_buffer.len() - (*capture_sink).pending_buffer_offset;
                let (data_chunk, rest) = data.split_at_mut(avail.min(data.len()));
                let (_, dest) = pending_buffer.split_at_mut((*capture_sink).pending_buffer_offset);
                let dest_len = dest.len();
                let dest = &mut dest[..data_chunk.len().min(dest_len)];
                data = rest;
                (*capture_sink).pending_buffer_offset += data_chunk.len();
                dest.copy_from_slice(data_chunk);
                if (*capture_sink).pending_buffer_offset == pending_buffer.len() {
                    let packet =
                        AudioPacket { data: (*capture_sink).pending_buffer.take().unwrap() };
                    (*capture_sink).pending_buffer_offset = 0;
                    if let Some(failed) = (*capture_sink).writer.try_push(packet) {
                        (*capture_sink).pending_buffer = Some(failed.data);
                        println!("xrun: dropped capture sink packet");
                    } else {
                        (*capture_sink).waiter.alert();
                    }
                }
            }
            capture_sink = (*capture_sink).next;
        }
    }
}

struct ControlThreadState {
    next_object_id: u64,
    request_rx: AtomicRingReader<AudioControlThreadRequest>,
    response_tx: AtomicRingWriter<AudioControlThreadResponse>,
    response_waiter: AtomicRingWaiter<AudioControlThreadResponse>,
    rt_heap: RtObjectHeap,
}

fn data_as_f32_slice(data: &mut pipewire::spa::data::Data) -> Option<&mut [f32]> {
    let ptr = data.as_raw();

    if ptr.data.is_null() {
        return None;
    }

    unsafe {
        let len = (*ptr.chunk).size as usize / std::mem::size_of::<f32>();
        let offset = (*ptr.chunk).offset as usize;
        Some(std::slice::from_raw_parts_mut(ptr.data.add(offset) as *mut f32, len))
    }
}

unsafe fn audio_control_thread_main(state: Rc<AudioControlThreadState>) -> anyhow::Result<()> {
    let main_loop = MainLoop::new()?;
    let context = Rc::new(Context::new(&main_loop)?);
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
                    let rt = &mut *state.rt.get();
                    if let Some(mut buffer) = stream.dequeue_buffer() {
                        for data in buffer.datas_mut() {
                            if let Some(samples) = data_as_f32_slice(data) {
                                rt.process_capture_stream(samples);
                            }
                        }
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
                let _rt = &mut *state.rt.get();
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
                data_buf.fill(0.0);
                (*data0.chunk).offset = 0;
                (*data0.chunk).stride = stride as i32;
                (*data0.chunk).size = (data_buf.len() * std::mem::size_of::<f32>()) as u32;
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
        let context = context.clone();
        move || {
            let ct = &mut *state.ct.get();
            while let Some(request) = ct.request_rx.try_pop() {
                match request {
                    AudioControlThreadRequest::AddCaptureSink { interval, waiter, writer } => {
                        let rate = (SAMPLE_RATE as f32 * interval.as_secs_f32()) as usize;
                        let buf_len = rate * NUM_CHANNELS;
                        let (buf_rx, mut buf_tx) = AtomicRing::new(CAPTURE_BUFFER_POOL_SIZE);
                        // TODO think through how to safely represent the sharing of data with
                        // the RT thread. RtPtr is just butchering around the problem.
                        let object_id = ct.next_object_id;
                        ct.next_object_id += 1;
                        let capture_sink = RtPtr(Box::into_raw(Box::new_in(
                            RtCaptureSink::new(object_id, buf_rx, waiter, writer),
                            ct.rt_heap.clone(),
                        )));
                        for _i in 0..CAPTURE_BUFFER_POOL_SIZE {
                            let buf = RtBuffer::new(buf_len, ct.rt_heap.clone());
                            assert!(buf_tx.try_push(buf).is_none());
                        }
                        state.invoke_rt(&context, move |rt| {
                            let capture_sink = capture_sink.into_raw();
                            (*capture_sink).next = rt.capture_sinks;
                            rt.capture_sinks = capture_sink;
                        });
                        assert!(ct
                            .response_tx
                            .try_push(AudioControlThreadResponse::CaptureStream {
                                object_id,
                                free_buffers: buf_tx,
                            })
                            .is_none());
                        ct.response_waiter.alert();
                    }
                    AudioControlThreadRequest::DestroyObject { object_id } => {
                        // TODO: we should use intrusive double linked lists of rt objects
                        // and an object table. then we can have O(1) removal
                        state.invoke_rt(&context, move |rt| {
                            let mut capture_sink = rt.capture_sinks;
                            let mut prev: *mut RtCaptureSink = ptr::null_mut();
                            while !capture_sink.is_null() {
                                if (*capture_sink).object_id == object_id {
                                    if prev.is_null() {
                                        rt.capture_sinks = (*capture_sink).next;
                                    } else {
                                        (*prev).next = (*capture_sink).next;
                                    }
                                    break;
                                }
                                prev = capture_sink;
                                capture_sink = (*capture_sink).next;
                            }
                        });
                        assert!(ct.response_tx.try_push(AudioControlThreadResponse::Ok).is_none());
                        ct.response_waiter.alert();
                    }
                }
            }
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
    // TODO: destroy all rt objects here
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

struct AudioControlThreadState {
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

impl AudioControlThreadState {
    // blocking invoke to RT thread. the callback can access the real-time state.
    // Unsafe for many reasons. You must ensure that invoke_rt is not re-entered,
    // otherwise you'll create a non-unique &mut as Pipewire will run the callback
    // directly on the invoking thread.
    pub unsafe fn invoke_rt<F>(&self, context: &Context<MainLoop>, f: F)
    where
        F: FnOnce(&mut RealTimeThreadState) + 'static + Send,
    {
        unsafe {
            unsafe extern "C" fn trampoline<F>(
                _loop: *mut libspa_sys::spa_loop, _is_async: bool, _seq: u32, _data: *const c_void,
                _size: usize, user_data: *mut c_void,
            ) -> i32
            where
                F: FnOnce(&mut RealTimeThreadState) + 'static + Send,
            {
                let (rt, f): &mut (*mut RealTimeThreadState, Option<F>) =
                    &mut *(user_data as *mut _);
                (f.take().unwrap())(&mut **rt);
                0
            }
            let data_loop = pipewire_sys::pw_context_get_data_loop(context.as_ptr());
            let mut user_data = (self.rt.get(), Some(f));
            pipewire_sys::pw_data_loop_invoke(
                data_loop,
                Some(trampoline::<F>),
                0,
                ptr::null_mut(),
                0,
                true,
                &mut user_data as *mut _ as *mut c_void,
            );
        }
    }
}

enum AudioControlThreadRequest {
    AddCaptureSink {
        interval: Duration,
        waiter: AtomicRingWaiter<AudioPacket>,
        writer: AtomicRingWriter<AudioPacket>,
    },
    DestroyObject {
        object_id: u64,
    },
}

enum AudioControlThreadResponse {
    Ok,
    Initialized {
        control_event: *mut libspa_sys::spa_source,
        quit_event: *mut libspa_sys::spa_source,
        pw_loop: *mut pipewire_sys::pw_loop,
    },
    InitializationFailed, // TODO wrap an error
    CaptureStream {
        object_id: u64,
        free_buffers: AtomicRingWriter<RtBuffer>,
    },
}

unsafe impl Send for AudioControlThreadResponse {}

struct AudioPacket {
    data: RtBuffer,
}

pub struct AudioStream {
    control_thread: AudioControlThread,
    object_id: u64,
    free_buffers: AtomicRingWriter<RtBuffer>,
    waiter: AtomicRingWaiter<AudioPacket>,
}

impl AudioStream {
    pub fn next(&mut self, buf: &mut [f32]) {
        let packet = self.waiter.wait_pop();
        let data = packet.data.as_slice();
        buf.copy_from_slice(data);
        assert!(self.free_buffers.try_push(packet.data).is_none());
    }
}

impl Drop for AudioStream {
    fn drop(&mut self) {
        self.control_thread.destroy_object(self.object_id);
    }
}

struct AudioControlThreadInner {
    thread: Option<std::thread::JoinHandle<()>>,
    waiter: AtomicRingWaiter<AudioControlThreadResponse>,
    request_tx: AtomicRingWriter<AudioControlThreadRequest>,
    control_event: *mut libspa_sys::spa_source,
    quit_event: *mut libspa_sys::spa_source,
    pw_loop: *mut pipewire_sys::pw_loop,
}

impl AudioControlThreadInner {
    fn try_send_request(&mut self, request: AudioControlThreadRequest) -> anyhow::Result<()> {
        if self.request_tx.try_push(request).is_some() {
            bail!("failed to send request to audio control thread");
        }
        unsafe { signal_event(self.pw_loop, self.control_event) }
        Ok(())
    }
}

impl Drop for AudioControlThreadInner {
    fn drop(&mut self) {
        unsafe { signal_event(self.pw_loop, self.quit_event) }
        self.thread.take().unwrap().join().unwrap();
    }
}

#[derive(Clone)]
pub struct AudioControlThread {
    inner: Arc<Mutex<AudioControlThreadInner>>,
}

impl AudioControlThread {
    pub fn new() -> anyhow::Result<Self> {
        // TODO: currently limited to a single outstanding request
        let (response_rx, response_tx) = AtomicRing::new(1);
        let (request_rx, request_tx) = AtomicRing::new(1);
        let waiter = AtomicRingWaiter::new(response_rx);
        let thread = Some(std::thread::spawn({
            let waiter = waiter.clone();
            move || unsafe {
                let state = Rc::new(AudioControlThreadState {
                    rt: UnsafeCell::new(RealTimeThreadState::new()),
                    ct: UnsafeCell::new(ControlThreadState {
                        next_object_id: 0,
                        request_rx,
                        response_tx,
                        response_waiter: waiter,
                        rt_heap: RtObjectHeap::new(1024 * 1024, 8, ()),
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
                Ok(Self {
                    inner: Arc::new(Mutex::new(AudioControlThreadInner {
                        thread,
                        waiter,
                        control_event,
                        quit_event,
                        pw_loop,
                        request_tx,
                    })),
                })
            }
            AudioControlThreadResponse::InitializationFailed => {
                bail!("failed to initialize audio control thread")
            }
            _ => unreachable!("invalid response"),
        }
    }

    pub fn capture_audio_stream(&self, interval: Duration) -> anyhow::Result<AudioStream> {
        let (response_rx, response_tx) = AtomicRing::new(CAPTURE_BUFFER_POOL_SIZE);
        let response_waiter = AtomicRingWaiter::new(response_rx);
        let request = AudioControlThreadRequest::AddCaptureSink {
            interval,
            waiter: response_waiter.clone(),
            writer: response_tx,
        };
        // TODO: once we have a request/response thing that can handle concurrent requests,
        // this lock can be dropped while waiting
        let mut inner = self.inner.lock();
        inner.try_send_request(request)?;
        match inner.waiter.wait_pop() {
            AudioControlThreadResponse::CaptureStream { object_id, free_buffers } => {
                Ok(AudioStream {
                    control_thread: self.clone(),
                    object_id,
                    free_buffers,
                    waiter: response_waiter,
                })
            }
            _ => unreachable!("invalid response"),
        }
    }

    fn destroy_object(&self, object_id: u64) {
        let request = AudioControlThreadRequest::DestroyObject { object_id };
        let mut inner = self.inner.lock();
        inner.try_send_request(request).unwrap();
        match inner.waiter.wait_pop() {
            AudioControlThreadResponse::Ok => {}
            _ => unreachable!("invalid response"),
        }
    }
}
