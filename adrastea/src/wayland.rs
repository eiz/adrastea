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
    borrow::BorrowMut,
    cell::RefCell,
    ffi::CStr,
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};
use std::os::{
    fd::{FromRawFd, OwnedFd},
    raw::c_void,
};

use alloc::{collections::BTreeMap, sync::Arc};
use memmap2::MmapMut;
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

use crate::rt_alloc::{ArenaAllocator, ArenaHandle, RtObjectHeap};

#[derive(Debug)]
struct WaylandArenaHandleInner {
    pool: wl_shm_pool::WlShmPool,
    _fd: OwnedFd,
    mapping: MmapMut,
}

impl Drop for WaylandArenaHandleInner {
    fn drop(&mut self) {
        self.pool.destroy();
    }
}

#[derive(Clone, Debug)]
struct WaylandArenaHandle(Arc<WaylandArenaHandleInner>, usize);

impl ArenaHandle for WaylandArenaHandle {
    fn as_ptr(&self) -> *mut c_void {
        // TODO the &refs and the &muts are messed here
        unsafe { (self.0.mapping.as_ptr() as *mut c_void).add(self.1) }
    }

    unsafe fn offset(&self, offset: usize) -> Self {
        Self(self.0.clone(), self.1 + offset)
    }
}

#[derive(Clone, Debug)]
struct WaylandArenaAllocator<T>(PhantomData<T>);

impl<T: Dispatch<wl_shm_pool::WlShmPool, ()> + 'static> ArenaAllocator
    for WaylandArenaAllocator<T>
{
    type Handle = WaylandArenaHandle;
    type Params = (wl_shm::WlShm, &'static CStr, wayland_client::QueueHandle<T>);

    fn allocate(size: u32, params: &Self::Params) -> Self::Handle {
        unsafe {
            let memfd = libc::memfd_create(params.1.as_ptr(), 0);
            assert!(memfd >= 0);
            assert_eq!(libc::ftruncate(memfd, size as _), 0);
            let mapping = MmapMut::map_mut(memfd).unwrap();
            // TODO we need to use F_SEAL_FUTURE_WRITE here
            let inner = Arc::new(WaylandArenaHandleInner {
                pool: params.0.create_pool(memfd, size as i32, &params.2, ()),
                _fd: OwnedFd::from_raw_fd(memfd),
                mapping,
            });
            WaylandArenaHandle(inner, 0)
        }
    }

    unsafe fn deallocate(_handle: &Self::Handle, _size: u32, _params: &Self::Params) {
        // the buffer pool is released automatically when the handle goes out of scope
    }
}

#[derive(Debug)]
struct WaylandShmBuffer {
    handle: WaylandArenaHandle,
    buffer: wl_buffer::WlBuffer,
}

impl WaylandShmBuffer {
    pub fn new(handle: WaylandArenaHandle, buffer: wl_buffer::WlBuffer) -> Self {
        Self { handle, buffer }
    }
}

impl Drop for WaylandShmBuffer {
    fn drop(&mut self) {
        self.buffer.destroy();
    }
}

#[derive(Debug)]
struct WaylandShmAllocator<T>
where
    T: 'static,
    T: Dispatch<wl_shm_pool::WlShmPool, ()>,
    T: Dispatch<wl_buffer::WlBuffer, ()>,
{
    heap: RtObjectHeap<WaylandArenaAllocator<T>>,
}

impl<T> WaylandShmAllocator<T>
where
    T: 'static,
    T: Dispatch<wl_shm_pool::WlShmPool, ()>,
    T: Dispatch<wl_buffer::WlBuffer, ()>,
{
    pub fn new(heap: RtObjectHeap<WaylandArenaAllocator<T>>) -> Self {
        Self { heap }
    }

    pub fn create_buffer(
        &self, width: i32, height: i32, stride: i32, format: wl_shm::Format,
        qhandle: &wayland_client::QueueHandle<T>,
    ) -> WaylandShmBuffer {
        println!("width: {}, height: {}, stride: {}", width, height, stride);
        let total_size = height.checked_mul(stride).unwrap();
        let handle = self
            .heap
            .allocate_handle(Layout::from_size_align(total_size as usize, 32).unwrap())
            .unwrap();
        let buffer = handle.0.pool.create_buffer(
            handle.1 as i32,
            width,
            height,
            stride,
            format,
            qhandle,
            (),
        );
        WaylandShmBuffer::new(handle, buffer)
    }
}

#[derive(Debug, Eq, PartialEq, Copy, Clone, Ord, PartialOrd)]
struct DelegateId(u64);

impl DelegateId {
    pub fn new() -> Self {
        static mut ID: AtomicU64 = AtomicU64::new(0);
        Self(unsafe { ID.fetch_add(1, Ordering::SeqCst) })
    }
}

impl From<u64> for DelegateId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

trait IWindow {
    fn frame_callback(&self, qhandle: &wayland_client::QueueHandle<WaylandTest>);
    fn finish_configure(&self, qhandle: &wayland_client::QueueHandle<WaylandTest>);
}

trait ITopLevelWindow {
    fn toplevel_configure(
        &self, alloc: &WaylandShmAllocator<WaylandTest>,
        qhandle: &wayland_client::QueueHandle<WaylandTest>, width: i32, height: i32,
    );
}

#[derive(Debug)]
struct TopLevelWindowInner {
    id: DelegateId,
    width: i32,
    height: i32,
    frame_number: i64,
    surface: wl_surface::WlSurface,
    xdg_surface: xdg_surface::XdgSurface,
    xdg_toplevel: xdg_toplevel::XdgToplevel,
    buffer: Option<WaylandShmBuffer>,
    frame_callback: Option<wl_callback::WlCallback>,
}

#[derive(Debug)]
struct TopLevelWindow(RefCell<TopLevelWindowInner>);

impl IWindow for TopLevelWindow {
    fn finish_configure(&self, qhandle: &wayland_client::QueueHandle<WaylandTest>) {
        let mut inner = self.0.borrow_mut();
        inner.surface.attach(inner.buffer.as_ref().map(|b| &b.buffer), 0, 0);
        if inner.frame_callback.is_none() {
            inner.frame_callback = Some(inner.surface.frame(qhandle, inner.id));
        }
        inner.surface.commit();
    }

    fn frame_callback(&self, qhandle: &wayland_client::QueueHandle<WaylandTest>) {
        let mut inner = self.0.borrow_mut();
        inner.frame_callback = None;
        inner.frame_number += 1;
        // it feels silly even putting TODOs in this code but...
        // we definitely need a proper swap chain abstraction to manage in-use/free
        // buffers.
        let color = if inner.frame_number % 2 == 0 { 0xFFFFFFFF } else { 0xFF000000 };
        let buffer = inner.buffer.as_ref().unwrap();
        // TODO bad bad bad
        let (ptr, len) = (buffer.handle.as_ptr(), inner.width * inner.height);
        let pixels = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, len as usize) };
        pixels.fill(color);
        inner.surface.attach(Some(&buffer.buffer), 0, 0);
        inner.frame_callback = Some(inner.surface.frame(&qhandle, inner.id));
        inner.surface.damage(0, 0, inner.width, inner.height);
        inner.surface.commit();
        return;
    }
}

impl ITopLevelWindow for TopLevelWindow {
    fn toplevel_configure(
        &self, alloc: &WaylandShmAllocator<WaylandTest>,
        qhandle: &wayland_client::QueueHandle<WaylandTest>, width: i32, height: i32,
    ) {
        let mut inner = self.0.borrow_mut();
        if width != 0 && height != 0 && (width != inner.width || height != inner.height) {
            let old_buffer = inner.buffer.take().unwrap();
            let buffer =
                alloc.create_buffer(width, height, width * 4, wl_shm::Format::Abgr8888, qhandle);
            old_buffer.buffer.destroy();
            let pixels = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer.handle.as_ptr() as *mut u32,
                    (height * width) as usize,
                )
            };
            pixels[..width as usize * height as usize].fill(0xFF00FF00);
            inner.buffer = Some(buffer);
            inner.height = height;
            inner.width = width;
        }
    }
}

impl IUnknown for TopLevelWindow {}
impl Provider for TopLevelWindow {
    fn provide<'a>(&'a self, demand: &mut std::any::Demand<'a>) {
        demand.provide_ref::<dyn IWindow>(self);
        demand.provide_ref::<dyn ITopLevelWindow>(self);
    }
}

trait IUnknown: Provider + core::fmt::Debug {}

impl dyn IUnknown + '_ {
    pub fn query_interface<T: ?Sized + 'static>(&self) -> Option<&T> {
        request_ref(self)
    }
}

#[derive(Default, Debug)]
struct WaylandTest {
    initialized: bool,
    post_bind_sync_complete: bool,
    delegate_table: BTreeMap<DelegateId, Box<dyn IUnknown>>,
    compositor: Option<wl_compositor::WlCompositor>,
    xdg_wm_base: Option<xdg_wm_base::XdgWmBase>,
    shm_allocator: Option<WaylandShmAllocator<Self>>,
    wl_shm: Option<wl_shm::WlShm>,
    wl_seat: Option<wl_seat::WlSeat>,
    wl_pointer: Option<wl_pointer::WlPointer>,
    _wl_keyboard: Option<wl_keyboard::WlKeyboard>,
    zwp_linux_dmabuf_v1: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    zwlr_layer_shell_v1: Option<zwlr_layer_shell_v1::ZwlrLayerShellV1>,
    layer_surface: Option<wl_surface::WlSurface>,
    wlr_layer_surface: Option<zwlr_layer_surface_v1::ZwlrLayerSurfaceV1>,
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

impl Dispatch<wl_surface::WlSurface, DelegateId> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &wl_surface::WlSurface, event: wl_surface::Event,
        _data: &DelegateId, _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
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

impl Dispatch<xdg_surface::XdgSurface, DelegateId> for WaylandTest {
    fn event(
        state: &mut Self, proxy: &xdg_surface::XdgSurface,
        event: <xdg_surface::XdgSurface as wayland_client::Proxy>::Event, data: &DelegateId,
        _conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("xdg_surface {:?}", event);
        match event {
            xdg_surface::Event::Configure { serial } => {
                proxy.ack_configure(serial);
                if let Some(delegate) = state.delegate_table.get_mut(data) {
                    let cw = delegate.query_interface::<dyn IWindow>().unwrap();
                    cw.finish_configure(qhandle);
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, DelegateId> for WaylandTest {
    fn event(
        state: &mut Self, _proxy: &xdg_toplevel::XdgToplevel,
        event: <xdg_toplevel::XdgToplevel as wayland_client::Proxy>::Event, data: &DelegateId,
        _conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("xdg_toplevel {:?}", event);
        match event {
            xdg_toplevel::Event::Configure { width, height, states: _ } => {
                if let Some(delegate) = state.delegate_table.get_mut(data) {
                    let top = delegate.query_interface::<dyn ITopLevelWindow>().unwrap();
                    top.toplevel_configure(
                        state.shm_allocator.as_ref().unwrap(),
                        qhandle,
                        width,
                        height,
                    );
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

impl Dispatch<wl_callback::WlCallback, DelegateId> for WaylandTest {
    fn event(
        state: &mut Self, proxy: &wl_callback::WlCallback,
        event: <wl_callback::WlCallback as wayland_client::Proxy>::Event, data: &DelegateId,
        conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        if let Some(delegate) = state.delegate_table.get_mut(data) {
            let cb = delegate.query_interface::<dyn IWindow>().unwrap();
            cb.frame_callback(qhandle);
        }
    }
}

impl Dispatch<wl_callback::WlCallback, ()> for WaylandTest {
    fn event(
        state: &mut Self, proxy: &wl_callback::WlCallback,
        _event: <wl_callback::WlCallback as wayland_client::Proxy>::Event, _data: &(),
        conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
    ) {
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
            let shm_allocator = WaylandShmAllocator::new(RtObjectHeap::new(
                4096 * 4096 * 4,
                8,
                (wl_shm.clone(), cstr::cstr!("wl_shm_pool"), qhandle.clone()),
            ));
            let pointer = wl_seat.get_pointer(qhandle, ());
            let _keyboard = wl_seat.get_keyboard(qhandle, ());

            let tlw_delegate = DelegateId::new();
            let surface = compositor.create_surface(qhandle, tlw_delegate);
            if let Some(dmabuf_api) = state.zwp_linux_dmabuf_v1.as_ref() {
                dmabuf_api.get_surface_feedback(&surface, qhandle, tlw_delegate);
            }
            let xdg_surface = xdg_wm_base.get_xdg_surface(&surface, qhandle, tlw_delegate);
            let xdg_top_level = xdg_surface.get_toplevel(qhandle, tlw_delegate);
            let buffer = shm_allocator.create_buffer(
                1920,
                1080,
                1920 * 4,
                wl_shm::Format::Abgr8888,
                qhandle,
            );

            xdg_top_level.set_title("Bruh".into());
            surface.commit();
            state.delegate_table.insert(
                tlw_delegate,
                Box::new(TopLevelWindow(RefCell::new(TopLevelWindowInner {
                    id: tlw_delegate,
                    width: 0,
                    height: 0,
                    frame_number: 0,
                    surface,
                    xdg_surface,
                    xdg_toplevel: xdg_top_level,
                    buffer: Some(buffer),
                    frame_callback: None,
                }))),
            );

            if let Some(zwlr_layer_shell_v1) = state.zwlr_layer_shell_v1.as_ref() {
                let layer_id = DelegateId::new();
                let layer_surface = compositor.create_surface(qhandle, layer_id);
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
            state.wl_pointer = Some(pointer);
            state.shm_allocator = Some(shm_allocator);
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

impl Dispatch<zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1, DelegateId> for WaylandTest {
    fn event(
        _state: &mut Self, _proxy: &zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1,
        event: <zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1 as wayland_client::Proxy>::Event,
        _data: &DelegateId, _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
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
        state: &mut Self, proxy: &zwlr_layer_surface_v1::ZwlrLayerSurfaceV1,
        event: <zwlr_layer_surface_v1::ZwlrLayerSurfaceV1 as wayland_client::Proxy>::Event,
        _data: &(), _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("zwlr_layer_surface_v1 {:?}", event);
        match event {
            zwlr_layer_surface_v1::Event::Configure { serial, width: _, height: _ } => {
                proxy.ack_configure(serial);
                state.layer_surface.as_ref().unwrap().commit();
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
