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

use std::os::fd::RawFd;

use wayland_client::{
    protocol::{
        wl_buffer, wl_callback, wl_compositor,
        wl_keyboard::{self, KeymapFormat},
        wl_pointer, wl_registry, wl_seat, wl_shm, wl_shm_pool, wl_surface,
    },
    Connection, Dispatch, WEnum,
};
use wayland_protocols::{
    wp::linux_dmabuf::zv1::client::{zwp_linux_dmabuf_feedback_v1, zwp_linux_dmabuf_v1},
    xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base},
};
use wayland_protocols_wlr::layer_shell::v1::client::{zwlr_layer_shell_v1, zwlr_layer_surface_v1};

#[derive(Default, Debug)]
struct WaylandTest {
    initialized: bool,
    post_bind_sync_complete: bool,
    frame_number: i64,
    width: i32,
    height: i32,
    compositor: Option<wl_compositor::WlCompositor>,
    xdg_wm_base: Option<xdg_wm_base::XdgWmBase>,
    wl_shm: Option<wl_shm::WlShm>,
    wl_shm_pool: Option<wl_shm_pool::WlShmPool>,
    surface: Option<wl_surface::WlSurface>,
    xdg_surface: Option<xdg_surface::XdgSurface>,
    xdg_top_level: Option<xdg_toplevel::XdgToplevel>,
    buffer: Option<wl_buffer::WlBuffer>,
    memfd: Option<RawFd>,
    mmap_mut: Option<memmap2::MmapMut>,
    wl_seat: Option<wl_seat::WlSeat>,
    wl_pointer: Option<wl_pointer::WlPointer>,
    _wl_keyboard: Option<wl_keyboard::WlKeyboard>,
    zwp_linux_dmabuf_v1: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    zwlr_layer_shell_v1: Option<zwlr_layer_shell_v1::ZwlrLayerShellV1>,
    layer_surface: Option<wl_surface::WlSurface>,
    wlr_layer_surface: Option<zwlr_layer_surface_v1::ZwlrLayerSurfaceV1>,
    frame_callback: Option<wl_callback::WlCallback>,
}

impl Dispatch<wl_registry::WlRegistry, ()> for WaylandTest {
    fn event(
        state: &mut Self, proxy: &wl_registry::WlRegistry, event: wl_registry::Event, _data: &(),
        _conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            println!("global {:?} {:?} {:?}", name, interface, version);
            if interface == "wl_compositor" {
                state.compositor = Some(proxy.bind(name, version, qhandle, ()));
            } else if interface == "xdg_wm_base" {
                state.xdg_wm_base = Some(proxy.bind(name, version, qhandle, ()));
            } else if interface == "wl_shm" {
                state.wl_shm = Some(proxy.bind(name, version, qhandle, ()));
            } else if interface == "wl_seat" {
                state.wl_seat = Some(proxy.bind(name, version, qhandle, ()));
            } else if interface == "zwp_linux_dmabuf_v1" {
                state.zwp_linux_dmabuf_v1 = Some(proxy.bind(name, version, qhandle, ()));
            } else if interface == "zwlr_layer_shell_v1" {
                state.zwlr_layer_shell_v1 = Some(proxy.bind(name, version, qhandle, ()));
            }
        }
    }
}

impl Dispatch<wl_compositor::WlCompositor, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &wl_compositor::WlCompositor, _event: wl_compositor::Event,
        _data: &(), _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        // we don't expect any events
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for WaylandTest {
    fn event(
        _state: &mut Self, proxy: &xdg_wm_base::XdgWmBase, event: xdg_wm_base::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        match event {
            xdg_wm_base::Event::Ping { serial } => proxy.pong(serial),
            _ => {}
        }
        println!("xdg_wm_base {:?}", event);
    }
}

impl Dispatch<wl_surface::WlSurface, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &wl_surface::WlSurface, event: wl_surface::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        match event {
            wl_surface::Event::Enter { .. } => todo!(),
            wl_surface::Event::Leave { .. } => todo!(),
            wl_surface::Event::PreferredBufferScale { .. } => todo!(),
            wl_surface::Event::PreferredBufferTransform { .. } => todo!(),
            _ => {}
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for WaylandTest {
    fn event(
        state: &mut Self, proxy: &xdg_surface::XdgSurface,
        event: <xdg_surface::XdgSurface as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("xdg_surface {:?}", event);
        match event {
            xdg_surface::Event::Configure { serial } => {
                proxy.ack_configure(serial);
                let surface = state.surface.as_ref().unwrap();
                surface.attach(state.buffer.as_ref(), 0, 0);
                if state.frame_callback.is_none() {
                    state.frame_callback = Some(surface.frame(&qhandle, ()));
                }
                surface.commit();
            }
            _ => {}
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for WaylandTest {
    fn event(
        state: &mut Self, _proxy: &xdg_toplevel::XdgToplevel,
        event: <xdg_toplevel::XdgToplevel as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("xdg_toplevel {:?}", event);
        match event {
            xdg_toplevel::Event::Configure { width, height, states: _ } => {
                if width != 0 && height != 0 && (width != state.width || height != state.height) {
                    let old_buffer = state.buffer.take().unwrap();
                    let wl_shm_pool = state.wl_shm_pool.as_ref().unwrap();
                    let buffer = wl_shm_pool.create_buffer(
                        0,
                        width,
                        height,
                        width * 4,
                        wl_shm::Format::Abgr8888,
                        qhandle,
                        (),
                    );
                    old_buffer.destroy();
                    let mapping = state.mmap_mut.as_mut().unwrap();
                    let pixels = unsafe {
                        std::slice::from_raw_parts_mut(
                            mapping.as_mut_ptr() as *mut u32,
                            mapping.len() / 4,
                        )
                    };
                    pixels[..width as usize * height as usize].fill(0xFF00FF00);
                    state.buffer = Some(buffer);
                    state.height = height;
                    state.width = width;
                }
            }
            xdg_toplevel::Event::Close => {}
            xdg_toplevel::Event::ConfigureBounds { width: _, height: _ } => {}
            xdg_toplevel::Event::WmCapabilities { capabilities: _ } => {}
            _ => {}
        }
    }
}

impl Dispatch<wl_shm::WlShm, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &wl_shm::WlShm,
        event: <wl_shm::WlShm as wayland_client::Proxy>::Event, _data: &(), _conn: &Connection,
        _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("wl_shm {:?}", event);
        //
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &wl_shm_pool::WlShmPool,
        event: <wl_shm_pool::WlShmPool as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("wl_shm_pool {:?}", event);
    }
}

impl Dispatch<wl_buffer::WlBuffer, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &wl_buffer::WlBuffer,
        _event: <wl_buffer::WlBuffer as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        // println!("wl_buffer {:?}", event);
    }
}

impl Dispatch<wl_pointer::WlPointer, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &wl_pointer::WlPointer,
        event: <wl_pointer::WlPointer as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("wl_pointer {:?}", event);
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &wl_keyboard::WlKeyboard,
        event: <wl_keyboard::WlKeyboard as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("wl_keyboard {:?}", event);
        match event {
            wl_keyboard::Event::Keymap {
                format: WEnum::Value(KeymapFormat::XkbV1),
                fd: _,
                size: _,
            } => {
                //let mmap = unsafe { memmap2::Mmap::map(&fd).unwrap() };
                //let xkb_text = CStr::from_bytes_with_nul(&mmap).unwrap();
                //println!("{}", xkb_text.to_string_lossy());
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &wl_seat::WlSeat,
        event: <wl_seat::WlSeat as wayland_client::Proxy>::Event, _data: &(), _conn: &Connection,
        _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("wl_seat {:?}", event);
    }
}

impl Dispatch<wl_callback::WlCallback, ()> for WaylandTest {
    fn event(
        state: &mut Self, proxy: &wl_callback::WlCallback,
        _event: <wl_callback::WlCallback as wayland_client::Proxy>::Event, _data: &(),
        conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        if Some(proxy) == state.frame_callback.as_ref() {
            state.frame_callback = None;
            state.frame_number += 1;
            let surface = state.surface.as_ref().unwrap();
            // it feels silly even putting TODOs in this code but...
            // we definitely need a proper swap chain abstraction to manage in-use/free
            // buffers.
            let color = if state.frame_number % 2 == 0 { 0xFFFFFFFF } else { 0xFF000000 };
            let mapping = state.mmap_mut.as_mut().unwrap();
            let (ptr, len) = (mapping.as_mut_ptr(), mapping.len());
            drop(mapping);
            let pixels = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, len / 4) };
            pixels.fill(color);
            state.frame_callback = Some(surface.frame(&qhandle, ()));
            surface.attach(state.buffer.as_ref(), 0, 0);
            surface.damage(0, 0, state.width, state.height);
            surface.commit();
            return;
        }
        if !state.post_bind_sync_complete {
            state.post_bind_sync_complete = true;
            conn.display().sync(qhandle, ());
            return;
        }
        // TODO: theres different kinds of wl_callbacks, don't just assume 🤣
        if !state.initialized
            && state.compositor.is_some()
            && state.xdg_wm_base.is_some()
            && state.wl_shm.is_some()
            && state.wl_seat.is_some()
        {
            let compositor = state.compositor.as_ref().unwrap();
            let xdg_wm_base = state.xdg_wm_base.as_ref().unwrap();
            let wl_shm = state.wl_shm.as_ref().unwrap();
            let wl_seat = state.wl_seat.as_ref().unwrap();
            let surface = compositor.create_surface(qhandle, ());
            let xdg_surface = xdg_wm_base.get_xdg_surface(&surface, qhandle, ());
            let xdg_top_level = xdg_surface.get_toplevel(qhandle, ());
            let pointer = wl_seat.get_pointer(qhandle, ());
            let _keyboard = wl_seat.get_keyboard(qhandle, ());
            if let Some(dmabuf_api) = state.zwp_linux_dmabuf_v1.as_ref() {
                dmabuf_api.get_surface_feedback(&surface, qhandle, ());
            }
            let memfd = unsafe { libc::memfd_create(b"wl_shm_pool\0".as_ptr() as *const i8, 0) };
            unsafe {
                libc::ftruncate(memfd, 4096 * 4096 * 4);
                let mut mapping = memmap2::Mmap::map(memfd).unwrap().make_mut().unwrap();
                mapping.as_mut().fill(0xFF);
                state.mmap_mut = Some(mapping);
            }
            state.memfd = Some(memfd);

            let wl_shm_pool = wl_shm.create_pool(memfd, 4096 * 4096 * 4, qhandle, ());

            let buffer = wl_shm_pool.create_buffer(
                0,
                1920,
                1080,
                1920 * 4,
                wl_shm::Format::Abgr8888,
                qhandle,
                (),
            );

            xdg_top_level.set_title("Bruh".into());
            surface.commit();

            if let Some(zwlr_layer_shell_v1) = state.zwlr_layer_shell_v1.as_ref() {
                let layer_surface = compositor.create_surface(qhandle, ());
                let wlr_layer_surface = zwlr_layer_shell_v1.get_layer_surface(
                    &layer_surface,
                    None,
                    zwlr_layer_shell_v1::Layer::Overlay,
                    "bruh".into(),
                    qhandle,
                    (),
                );

                wlr_layer_surface.set_anchor(
                    zwlr_layer_surface_v1::Anchor::Bottom
                        | zwlr_layer_surface_v1::Anchor::Right
                        | zwlr_layer_surface_v1::Anchor::Left
                        | zwlr_layer_surface_v1::Anchor::Top,
                );
                layer_surface.commit();
                state.layer_surface = Some(layer_surface);
                state.wlr_layer_surface = Some(wlr_layer_surface);
            }

            state.surface = Some(surface);
            state.xdg_surface = Some(xdg_surface);
            state.xdg_top_level = Some(xdg_top_level);
            state.wl_shm_pool = Some(wl_shm_pool);
            state.buffer = Some(buffer);
            state.wl_pointer = Some(pointer);
            state.initialized = true;
        }
    }
}

impl Dispatch<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1,
        event: <zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1 as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("zwp_linux_dmabuf_v1 {:?}", event);
    }
}

impl Dispatch<zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1,
        event: <zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1 as wayland_client::Proxy>::Event,
        _data: &(), _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("zwp_linux_dmabuf_feedback {:?}", event);
    }
}

impl Dispatch<zwlr_layer_shell_v1::ZwlrLayerShellV1, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &zwlr_layer_shell_v1::ZwlrLayerShellV1,
        event: <zwlr_layer_shell_v1::ZwlrLayerShellV1 as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("zwlr_layer_shell_v1 {:?}", event);
    }
}

impl Dispatch<zwlr_layer_surface_v1::ZwlrLayerSurfaceV1, ()> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &zwlr_layer_surface_v1::ZwlrLayerSurfaceV1,
        event: <zwlr_layer_surface_v1::ZwlrLayerSurfaceV1 as wayland_client::Proxy>::Event,
        _data: &(), _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("zwlr_layer_surface_v1 {:?}", event);
        match event {
            zwlr_layer_surface_v1::Event::Configure { serial: _, width: _, height: _ } => {
                //
            }
            zwlr_layer_surface_v1::Event::Closed => {
                //
            }
            _ => {}
        }
    }
}

pub fn wayland_test() -> anyhow::Result<()> {
    let conn = Connection::connect_to_env()?;
    let display = conn.display();
    let mut event_queue = conn.new_event_queue();
    let handle = event_queue.handle();
    let _registry = display.get_registry(&handle, ());
    display.sync(&handle, ());
    println!("conn {:?}", conn);
    let mut state = WaylandTest::default();
    loop {
        event_queue.blocking_dispatch(&mut state)?;
    }
}
