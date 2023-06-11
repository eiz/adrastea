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

use core::{cell::UnsafeCell, ffi::c_void};
use std::f64::consts::PI;

use alloc::sync::Arc;
use libspa_sys::{
    spa_audio_info_raw, spa_format_audio_raw_build, spa_pod_builder, spa_pod_builder_init,
    SPA_PARAM_EnumFormat, SPA_AUDIO_FORMAT_F32,
};
use pipewire::{
    properties,
    spa::Direction,
    stream::{ListenerBuilderT, Stream, StreamFlags},
    Context, MainLoop,
};

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

struct AudioControlThread {
    thread: Option<std::thread::JoinHandle<()>>,
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

unsafe fn audio_control_thread_main() -> anyhow::Result<()> {
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
    main_loop.run();
    Ok(())
}

impl AudioControlThread {
    pub fn new() -> Self {
        let thread = Some(std::thread::spawn(|| unsafe {
            let _ = audio_control_thread_main(); // TODO handle error
        }));

        Self { thread }
    }
}

impl Drop for AudioControlThread {
    fn drop(&mut self) {
        self.thread.take().unwrap().join().unwrap();
    }
}

pub fn test() -> anyhow::Result<()> {
    let audio_control = AudioControlThread::new();
    Ok(())
}
