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

use core::ffi::c_void;

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

pub unsafe fn test() -> anyhow::Result<()> {
    let main_loop = MainLoop::new()?;
    let context = Context::new(&main_loop)?;
    let core = context.connect(None)?;
    let registry = core.get_registry()?;

    let _listener =
        registry.add_listener_local().global(|g| println!("global: {:?}", g)).register();
    let mut stream: Stream<()> = Stream::new(
        &core,
        "audio-capture",
        properties! {
            *pipewire::keys::MEDIA_TYPE => "Audio",
            *pipewire::keys::MEDIA_CATEGORY => "Capture",
            *pipewire::keys::MEDIA_ROLE => "Communication",
        },
    )?;
    let mut builder: spa_pod_builder = std::mem::zeroed();
    let mut buf = [0u8; 1024];
    spa_pod_builder_init(&mut builder, buf.as_mut_ptr() as *mut c_void, buf.len() as u32);
    let pod = spa_format_audio_raw_build(
        &mut builder,
        SPA_PARAM_EnumFormat,
        &mut spa_audio_info_raw { format: SPA_AUDIO_FORMAT_F32, ..std::mem::zeroed() },
    );
    let _stream_listener = stream
        .add_local_listener()
        .process(|_, _| {
            println!("process");
        })
        .state_changed(|_, _| {
            println!("state changed");
        })
        .register()?;
    stream.connect(
        Direction::Input,
        None,
        StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
        &mut [pod],
    )?;
    main_loop.run();
    Ok(())
}
