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
    fmt::Formatter,
    mem::MaybeUninit,
    ptr::{self},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use std::f64::consts::PI;

use alloc::{rc::Rc, sync::Arc};
use allocator_api2::boxed::Box;
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

use crate::{
    rt_alloc::RtObjectHeap,
    util::{AtomicRing, AtomicRingReader, AtomicRingWaiter, AtomicRingWriter},
};

const SAMPLE_RATE: u32 = 48000;
const NUM_CHANNELS: usize = 2;
const VOLUME: f32 = 0.1;
const CAPTURE_BUFFER_POOL_SIZE: usize = 16;

struct RealTimeThreadState {
    accumulator: f64,
    did_thing: bool,
    did_other_thing: bool,
    capture_sinks: *mut RtCaptureSink,
}

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

// TODO I feel like I'm definitely dupe'ing SPA/pipewire infra here
// lets revisit this and see if there's a more congruent way to do everything. but xplat
// also will def be a thing so...
struct RtBuffer(*mut [f32]);

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

impl AudioControlThreadInner {
    pub fn invoke_rt<F>(&self, context: &Context<MainLoop>, f: F)
    where
        F: Fn(&mut RealTimeThreadState) + 'static + Send,
    {
        unsafe {
            unsafe extern "C" fn invoke_data_loop_sync_trampoline<F>(
                _loop: *mut libspa_sys::spa_loop, _is_async: bool, _seq: u32, _data: *const c_void,
                _size: usize, user_data: *mut c_void,
            ) -> i32
            where
                F: Fn(&mut RealTimeThreadState) + 'static + Send,
            {
                let (rt, f): &(*mut RealTimeThreadState, *const F) = &*(user_data as *const _);
                (**f)(&mut **rt);
                0
            }
            let data_loop = pipewire_sys::pw_context_get_data_loop(context.as_ptr());
            let user_data = (self.rt.get(), &f as *const F);
            pipewire_sys::pw_data_loop_invoke(
                data_loop,
                Some(invoke_data_loop_sync_trampoline::<F>),
                0,
                ptr::null_mut(),
                0,
                true,
                &user_data as *const _ as *mut c_void,
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
                    // much yo, very lo
                    let rt = &mut *state.rt.get();
                    rt.did_thing = true;
                    if let Some(mut buffer) = stream.dequeue_buffer() {
                        for _data in buffer.datas_mut() {
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
        let context = context.clone();
        move || {
            let ct = &mut *state.ct.get();
            while let Some(request) = ct.request_rx.try_pop() {
                match request {
                    AudioControlThreadRequest::AddCaptureSink {
                        interval,
                        waiter: _,
                        writer: _,
                    } => {
                        let rate = (SAMPLE_RATE as f32 * interval.as_secs_f32()) as usize;
                        let buf_len = rate * NUM_CHANNELS * std::mem::size_of::<f32>();
                        let (buf_rx, mut buf_tx) = AtomicRing::new(CAPTURE_BUFFER_POOL_SIZE);
                        let capture_sink = RtPtr(Box::into_raw(Box::new_in(
                            RtCaptureSink { next: ptr::null_mut(), free_buffers: buf_rx },
                            ct.rt_heap.clone(),
                        )));

                        for _i in 0..CAPTURE_BUFFER_POOL_SIZE {
                            let buf: Box<[MaybeUninit<f32>], RtObjectHeap> =
                                Box::new_zeroed_slice_in(buf_len, ct.rt_heap.clone());
                            let buf =
                                Box::<[f32], RtObjectHeap>::into_raw(unsafe { buf.assume_init() });
                            let buf = RtBuffer(buf);

                            assert!(buf_tx.try_push(buf).is_none());
                        }

                        state.invoke_rt(&context, move |rt| {
                            let capture_sink = capture_sink.into_raw();
                            (*capture_sink).next = rt.capture_sinks;
                            rt.capture_sinks = capture_sink;
                            println!("we in there {:?} {:?}", std::thread::current(), capture_sink);
                        });
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
        let _response = self.waiter.wait_pop();
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
    let _audio_stream = audio_control.capture_audio_stream(Duration::from_millis(10))?;
    println!("we got capture stream");
    std::thread::sleep(Duration::from_secs(5));
    Ok(())
}
