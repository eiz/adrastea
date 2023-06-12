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
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use std::f64::consts::PI;

use alloc::sync::Arc;
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

use crate::util::{AtomicRing, AtomicRingWaiter, AtomicRingWriter};

const SAMPLE_RATE: u32 = 48000;
const NUM_CHANNELS: usize = 2;
const VOLUME: f32 = 0.1;

struct RealTimeEngine {
    accumulator: f64,
    did_thing: bool,
    did_other_thing: bool,
}

struct AudioControlThreadInner {
    // must only be accessed in process callbacks or otherwise on the pw_data_loop
    rt: UnsafeCell<RealTimeEngine>,
}

#[derive(Clone)]
enum AudioControlThreadResponse {
    Initialized {
        control_event: *mut libspa_sys::spa_source,
        quit_event: *mut libspa_sys::spa_source,
        pw_loop: *mut pipewire_sys::pw_loop,
    },
    InitializationFailed, // TODO wrap an error
}

unsafe impl Send for AudioControlThreadResponse {}

struct AudioControlThread {
    thread: Option<std::thread::JoinHandle<()>>,
    waiter: AtomicRingWaiter<AudioControlThreadResponse>,
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

unsafe fn audio_control_thread_main(
    tx: &mut AtomicRingWriter<AudioControlThreadResponse>,
    waiter: AtomicRingWaiter<AudioControlThreadResponse>,
) -> anyhow::Result<()> {
    let state = Arc::new(AudioControlThreadInner {
        rt: UnsafeCell::new(RealTimeEngine {
            accumulator: 0.0,
            did_thing: false,
            did_other_thing: false,
        }),
    });
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
    let control_event = main_loop.add_event(|| {
        println!("event handler");
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

    assert_eq!(
        1,
        tx.try_pushn(&[AudioControlThreadResponse::Initialized {
            control_event: control_event.as_ptr(),
            quit_event: quit_event.as_ptr(),
            pw_loop: main_loop.as_raw() as *const _ as *mut _
        }])
    );
    waiter.alert();
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
        // adding dependencies for this ... thing
        let (act_rx, mut act_tx) = AtomicRing::new(1);
        let waiter = AtomicRingWaiter::new(act_rx);
        let thread = Some(std::thread::spawn({
            let waiter = waiter.clone();
            move || unsafe {
                if let Err(_e) = audio_control_thread_main(&mut act_tx, waiter) {
                    assert_eq!(
                        1,
                        act_tx.try_pushn(&[AudioControlThreadResponse::InitializationFailed])
                    );
                }
            }
        }));
        match waiter.wait_pop() {
            AudioControlThreadResponse::Initialized { control_event, quit_event, pw_loop } => {
                Ok(Self { thread, waiter, control_event, quit_event, pw_loop })
            }
            AudioControlThreadResponse::InitializationFailed => {
                bail!("failed to initialize audio control thread")
            }
        }
    }

    pub fn prove_we_can_send_events(&self) {
        unsafe { signal_event(self.pw_loop, self.control_event) }
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

    let audio_control = AudioControlThread::new()?;
    println!("we got audio control");
    audio_control.prove_we_can_send_events();
    std::thread::sleep(Duration::from_secs(5));
    Ok(())
}
