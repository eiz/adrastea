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

// TODO: implement the proxy, dmabufs

use core::{
    alloc::Layout,
    any::Provider,
    cell::RefCell,
    ffi::CStr,
    marker::PhantomData,
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
};
use std::{
    fs::File,
    io::{self, BufReader},
    os::{
        fd::{FromRawFd, OwnedFd},
        raw::c_void,
    },
    path::Path,
};

use adrastea_core::{
    net::UnixScmStream,
    rt_alloc::{ArenaAllocator, ArenaHandle, RtObjectHeap},
    util::{round_up, IUnknown},
};
use alloc::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};
use anyhow::bail;
use byteorder::{ByteOrder, NativeEndian};
use memmap2::MmapMut;
use serde::Deserialize;
use skia_safe::{AlphaType, Canvas, ColorType, ISize, ImageInfo, Surface};
use wayland_client::{
    protocol::{
        wl_buffer, wl_callback, wl_compositor,
        wl_keyboard::{self, KeymapFormat},
        wl_pointer, wl_registry, wl_seat, wl_shm, wl_shm_pool, wl_surface,
    },
    Connection, Dispatch, EventQueue, QueueHandle, WEnum,
};
use wayland_protocols::{
    wp::linux_dmabuf::zv1::client::{zwp_linux_dmabuf_feedback_v1, zwp_linux_dmabuf_v1},
    xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base},
};
use wayland_protocols_wlr::layer_shell::v1::client::{zwlr_layer_shell_v1, zwlr_layer_surface_v1};

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

    fn allocate(size: usize, params: &Self::Params) -> Self::Handle {
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

    unsafe fn deallocate(_handle: &Self::Handle, _size: usize, _params: &Self::Params) {
        // the buffer pool is released automatically when the handle goes out of scope
    }
}

#[derive(Debug)]
struct ShmBuffer<T>
where
    T: 'static,
    T: Dispatch<wl_shm_pool::WlShmPool, ()>,
    T: Dispatch<wl_buffer::WlBuffer, ()>,
{
    heap: Arc<RtObjectHeap<WaylandArenaAllocator<T>>>,
    layout: Layout,
    handle: WaylandArenaHandle,
    buffer: wl_buffer::WlBuffer,
}

impl<T> ShmBuffer<T>
where
    T: 'static,
    T: Dispatch<wl_shm_pool::WlShmPool, ()>,
    T: Dispatch<wl_buffer::WlBuffer, ()>,
{
    pub fn new(
        heap: Arc<RtObjectHeap<WaylandArenaAllocator<T>>>, layout: Layout,
        handle: WaylandArenaHandle, buffer: wl_buffer::WlBuffer,
    ) -> Self {
        Self { heap, layout, handle, buffer }
    }
}

impl<T> Drop for ShmBuffer<T>
where
    T: 'static,
    T: Dispatch<wl_shm_pool::WlShmPool, ()>,
    T: Dispatch<wl_buffer::WlBuffer, ()>,
{
    fn drop(&mut self) {
        self.buffer.destroy();
        unsafe { self.heap.deallocate_handle(self.handle.clone(), self.layout) }
    }
}

#[derive(Debug)]
struct ShmAllocator<T>
where
    T: 'static,
    T: Dispatch<wl_shm_pool::WlShmPool, ()>,
    T: Dispatch<wl_buffer::WlBuffer, ()>,
{
    heap: Arc<RtObjectHeap<WaylandArenaAllocator<T>>>,
}

impl<T> ShmAllocator<T>
where
    T: 'static,
    T: Dispatch<wl_shm_pool::WlShmPool, ()>,
    T: Dispatch<wl_buffer::WlBuffer, ()>,
{
    pub fn new(heap: RtObjectHeap<WaylandArenaAllocator<T>>) -> Self {
        Self { heap: Arc::new(heap) }
    }

    pub fn create_buffer(
        &self, width: i32, height: i32, stride: i32, format: wl_shm::Format,
        qhandle: &wayland_client::QueueHandle<T>,
    ) -> ShmBuffer<T> {
        let total_size = height.checked_mul(stride).unwrap();
        let layout = Layout::from_size_align(total_size as usize, 32).unwrap();
        let handle = self.heap.allocate_handle(layout).unwrap();
        let buffer = handle.0.pool.create_buffer(
            handle.1 as i32,
            width,
            height,
            stride,
            format,
            qhandle,
            (),
        );
        ShmBuffer::new(self.heap.clone(), layout, handle, buffer)
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

trait ISurface: IUnknown {
    fn frame_callback(&self, qhandle: &wayland_client::QueueHandle<SurfaceClient>);
    fn finish_configure(&self, qhandle: &wayland_client::QueueHandle<SurfaceClient>);
}

trait ITopLevelSurface: IUnknown {
    fn toplevel_configure(
        &self, alloc: &ShmAllocator<SurfaceClient>,
        qhandle: &wayland_client::QueueHandle<SurfaceClient>, width: i32, height: i32,
    );
}

pub trait ISkiaPaint: IUnknown {
    fn on_paint_skia(&self, canvas: &mut Canvas, width: f32, height: f32);
}

#[derive(Debug)]
struct TopLevelSurfaceInner {
    id: DelegateId,
    width: i32,
    height: i32,
    surface: wl_surface::WlSurface,
    buffer: ShmBuffer<SurfaceClient>,
    frame_callback: Option<wl_callback::WlCallback>,
}

#[derive(Debug)]
struct TopLevelSurface<T: IUnknown + 'static> {
    inner: RefCell<TopLevelSurfaceInner>,
    user_delegate: T,
}

impl<T: IUnknown + 'static> TopLevelSurface<T> {
    pub fn new(
        tlw_delegate: DelegateId, surface: wl_surface::WlSurface, buffer: ShmBuffer<SurfaceClient>,
        user_delegate: T,
    ) -> Self {
        Self {
            inner: RefCell::new(TopLevelSurfaceInner {
                id: tlw_delegate,
                width: 0,
                height: 0,
                surface,
                buffer,
                frame_callback: None,
            }),
            user_delegate,
        }
    }
}

impl<T: IUnknown + 'static> IUnknown for TopLevelSurface<T> {}
impl<T: IUnknown + 'static> Provider for TopLevelSurface<T> {
    fn provide<'a>(&'a self, demand: &mut std::any::Demand<'a>) {
        self.user_delegate.provide(demand);
        demand.provide_ref::<dyn ISurface>(self);
        demand.provide_ref::<dyn ITopLevelSurface>(self);
    }
}

impl<T: IUnknown + 'static> ISurface for TopLevelSurface<T> {
    fn finish_configure(&self, qhandle: &wayland_client::QueueHandle<SurfaceClient>) {
        let mut inner = self.inner.borrow_mut();
        inner.surface.attach(Some(&inner.buffer.buffer), 0, 0);
        if inner.frame_callback.is_none() && inner.width > 0 && inner.height > 0 {
            inner.frame_callback = Some(inner.surface.frame(qhandle, inner.id));
        }
        inner.surface.commit();
    }

    fn frame_callback(&self, qhandle: &wayland_client::QueueHandle<SurfaceClient>) {
        let mut inner = self.inner.borrow_mut();
        inner.frame_callback = None;
        if inner.width == 0 || inner.height == 0 {
            return;
        }

        if let Some(skia_paint) = (self as &dyn IUnknown).query_interface::<dyn ISkiaPaint>() {
            // it feels silly even putting TODOs in this code but...
            // we definitely need a proper swap chain abstraction to manage in-use/free
            // buffers.
            // TODO bad bad bad
            let (ptr, len) = (inner.buffer.handle.as_ptr(), inner.width * inner.height * 4);
            let pixels = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len as usize) };
            let image_info = ImageInfo::new(
                ISize::new(inner.width, inner.height),
                ColorType::RGBA8888,
                AlphaType::Premul,
                None,
            );
            let mut surface =
                Surface::new_raster_direct(&image_info, pixels, inner.width as usize * 4, None)
                    .unwrap();
            let canvas = surface.canvas();
            canvas.scale((1.5, 1.5));
            skia_paint.on_paint_skia(canvas, inner.width as f32 / 1.5, inner.height as f32 / 1.5);
        }

        inner.surface.attach(Some(&inner.buffer.buffer), 0, 0);
        inner.frame_callback = Some(inner.surface.frame(&qhandle, inner.id));
        inner.surface.damage(0, 0, inner.width, inner.height);
        inner.surface.commit();
        return;
    }
}

impl<T: IUnknown> ITopLevelSurface for TopLevelSurface<T> {
    fn toplevel_configure(
        &self, alloc: &ShmAllocator<SurfaceClient>,
        qhandle: &wayland_client::QueueHandle<SurfaceClient>, width: i32, height: i32,
    ) {
        let mut inner = self.inner.borrow_mut();
        if width != 0 && height != 0 && (width != inner.width || height != inner.height) {
            let buffer =
                alloc.create_buffer(width, height, width * 4, wl_shm::Format::Abgr8888, qhandle);
            let pixels = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer.handle.as_ptr() as *mut u32,
                    (height * width) as usize,
                )
            };
            pixels[..width as usize * height as usize].fill(0xFF000000);
            inner.buffer = buffer;
            inner.height = height;
            inner.width = width;
        }
    }
}

#[derive(Debug)]
pub struct SurfaceClient {
    initialized: bool,
    post_bind_sync_complete: bool,
    delegate_table: BTreeMap<DelegateId, Box<dyn IUnknown>>,
    handle: QueueHandle<Self>,
    compositor: Option<wl_compositor::WlCompositor>,
    xdg_wm_base: Option<xdg_wm_base::XdgWmBase>,
    shm_allocator: Option<ShmAllocator<Self>>,
    wl_shm: Option<wl_shm::WlShm>,
    wl_seat: Option<wl_seat::WlSeat>,
    wl_pointer: Option<wl_pointer::WlPointer>,
    _wl_keyboard: Option<wl_keyboard::WlKeyboard>,
    zwp_linux_dmabuf_v1: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    zwlr_layer_shell_v1: Option<zwlr_layer_shell_v1::ZwlrLayerShellV1>,
    layer_surface: Option<wl_surface::WlSurface>,
    wlr_layer_surface: Option<zwlr_layer_surface_v1::ZwlrLayerSurfaceV1>,
}

impl SurfaceClient {
    pub fn connect_to_env() -> anyhow::Result<(EventQueue<Self>, Self)> {
        let conn = Connection::connect_to_env()?;
        let display = conn.display();
        let mut event_queue = conn.new_event_queue();
        let handle = event_queue.handle();
        let _registry = display.get_registry(&handle, ());
        display.sync(&handle, ());
        let mut client = Self {
            initialized: false,
            post_bind_sync_complete: false,
            delegate_table: BTreeMap::new(),
            handle,
            compositor: None,
            xdg_wm_base: None,
            shm_allocator: None,
            wl_shm: None,
            wl_seat: None,
            wl_pointer: None,
            _wl_keyboard: None,
            zwp_linux_dmabuf_v1: None,
            zwlr_layer_shell_v1: None,
            layer_surface: None,
            wlr_layer_surface: None,
        };
        while !client.initialized {
            event_queue.blocking_dispatch(&mut client)?;
        }
        Ok((event_queue, client))
    }

    pub fn create_toplevel_surface<T: IUnknown + 'static>(&mut self, user_delegate: T) {
        assert!(self.initialized);
        let shm_allocator = self.shm_allocator.as_ref().unwrap();
        let compositor = self.compositor.as_ref().unwrap();
        let xdg_wm_base = self.xdg_wm_base.as_ref().unwrap();
        let tlw_delegate = DelegateId::new();
        let surface = compositor.create_surface(&self.handle, tlw_delegate);
        if let Some(dmabuf_api) = self.zwp_linux_dmabuf_v1.as_ref() {
            dmabuf_api.get_surface_feedback(&surface, &self.handle, tlw_delegate);
        }
        let xdg_surface = xdg_wm_base.get_xdg_surface(&surface, &self.handle, tlw_delegate);
        let xdg_top_level = xdg_surface.get_toplevel(&self.handle, tlw_delegate);
        let buffer = shm_allocator.create_buffer(
            1920,
            1080,
            1920 * 4,
            wl_shm::Format::Abgr8888,
            &self.handle,
        );

        xdg_top_level.set_title("Bruh".into());
        surface.commit();
        self.delegate_table.insert(
            tlw_delegate,
            Box::new(TopLevelSurface::new(tlw_delegate, surface, buffer, user_delegate)),
        );
    }
}

impl Dispatch<wl_registry::WlRegistry, ()> for SurfaceClient {
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

impl Dispatch<wl_compositor::WlCompositor, ()> for SurfaceClient {
    fn event(
        _state: &mut Self, _proxy: &wl_compositor::WlCompositor, _event: wl_compositor::Event,
        _data: &(), _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        // we don't expect any events
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for SurfaceClient {
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

impl Dispatch<wl_surface::WlSurface, DelegateId> for SurfaceClient {
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

impl Dispatch<xdg_surface::XdgSurface, DelegateId> for SurfaceClient {
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
                    let cw = delegate.query_interface::<dyn ISurface>().unwrap();
                    cw.finish_configure(qhandle);
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, DelegateId> for SurfaceClient {
    fn event(
        state: &mut Self, _proxy: &xdg_toplevel::XdgToplevel,
        event: <xdg_toplevel::XdgToplevel as wayland_client::Proxy>::Event, data: &DelegateId,
        _conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("xdg_toplevel {:?}", event);
        match event {
            xdg_toplevel::Event::Configure { width, height, states: _ } => {
                if let Some(delegate) = state.delegate_table.get_mut(data) {
                    let top = delegate.query_interface::<dyn ITopLevelSurface>().unwrap();
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

impl Dispatch<wl_shm::WlShm, ()> for SurfaceClient {
    fn event(
        _state: &mut Self, _proxy: &wl_shm::WlShm,
        event: <wl_shm::WlShm as wayland_client::Proxy>::Event, _data: &(), _conn: &Connection,
        _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("wl_shm {:?}", event);
        //
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for SurfaceClient {
    fn event(
        _state: &mut Self, _proxy: &wl_shm_pool::WlShmPool,
        event: <wl_shm_pool::WlShmPool as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("wl_shm_pool {:?}", event);
    }
}

impl Dispatch<wl_buffer::WlBuffer, ()> for SurfaceClient {
    fn event(
        _state: &mut Self, _proxy: &wl_buffer::WlBuffer,
        _event: <wl_buffer::WlBuffer as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        // println!("wl_buffer {:?}", event);
    }
}

impl Dispatch<wl_pointer::WlPointer, ()> for SurfaceClient {
    fn event(
        _state: &mut Self, _proxy: &wl_pointer::WlPointer,
        event: <wl_pointer::WlPointer as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("wl_pointer {:?}", event);
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for SurfaceClient {
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

impl Dispatch<wl_seat::WlSeat, ()> for SurfaceClient {
    fn event(
        _state: &mut Self, _proxy: &wl_seat::WlSeat,
        event: <wl_seat::WlSeat as wayland_client::Proxy>::Event, _data: &(), _conn: &Connection,
        _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("wl_seat {:?}", event);
    }
}

impl Dispatch<wl_callback::WlCallback, DelegateId> for SurfaceClient {
    fn event(
        state: &mut Self, _proxy: &wl_callback::WlCallback,
        _event: <wl_callback::WlCallback as wayland_client::Proxy>::Event, data: &DelegateId,
        _conn: &Connection, qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        if let Some(delegate) = state.delegate_table.get_mut(data) {
            let cb = delegate.query_interface::<dyn ISurface>().unwrap();
            cb.frame_callback(qhandle);
        }
    }
}

impl Dispatch<wl_callback::WlCallback, ()> for SurfaceClient {
    fn event(
        state: &mut Self, _proxy: &wl_callback::WlCallback,
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
            let _xdg_wm_base = state.xdg_wm_base.as_ref().unwrap();
            let wl_shm = state.wl_shm.as_ref().unwrap();
            let wl_seat = state.wl_seat.as_ref().unwrap();
            let shm_allocator = ShmAllocator::new(RtObjectHeap::new(
                4096 * 4096 * 4,
                8,
                (wl_shm.clone(), cstr::cstr!("wl_shm_pool"), qhandle.clone()),
            ));
            let pointer = wl_seat.get_pointer(qhandle, ());
            let _keyboard = wl_seat.get_keyboard(qhandle, ());

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

impl Dispatch<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1, ()> for SurfaceClient {
    fn event(
        _state: &mut Self, _proxy: &zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1,
        event: <zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1 as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("zwp_linux_dmabuf_v1 {:?}", event);
    }
}

impl Dispatch<zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1, DelegateId>
    for SurfaceClient
{
    fn event(
        _state: &mut Self, _proxy: &zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1,
        event: <zwp_linux_dmabuf_feedback_v1::ZwpLinuxDmabufFeedbackV1 as wayland_client::Proxy>::Event,
        _data: &DelegateId, _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("zwp_linux_dmabuf_feedback {:?}", event);
    }
}

impl Dispatch<zwlr_layer_shell_v1::ZwlrLayerShellV1, ()> for SurfaceClient {
    fn event(
        _state: &mut Self, _proxy: &zwlr_layer_shell_v1::ZwlrLayerShellV1,
        event: <zwlr_layer_shell_v1::ZwlrLayerShellV1 as wayland_client::Proxy>::Event, _data: &(),
        _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
    ) {
        println!("zwlr_layer_shell_v1 {:?}", event);
    }
}

impl Dispatch<zwlr_layer_surface_v1::ZwlrLayerSurfaceV1, ()> for SurfaceClient {
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

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq)]
pub enum WaylandDataType {
    #[serde(rename = "int")]
    Int,
    #[serde(rename = "uint")]
    Uint,
    #[serde(rename = "fixed")]
    Fixed,
    #[serde(rename = "string")]
    String,
    #[serde(rename = "object")]
    Object,
    #[serde(rename = "new_id")]
    NewId,
    #[serde(rename = "array")]
    Array,
    #[serde(rename = "fd")]
    Fd,
}

#[derive(Debug, Deserialize)]
pub struct WaylandDescription {
    #[serde(rename = "@summary")]
    pub summary: Option<String>,
    #[serde(rename = "$text")]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct WaylandArg {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@type")]
    pub data_type: WaylandDataType,
    #[serde(rename = "@summary")]
    pub summary: Option<String>,
    #[serde(rename = "@interface")]
    pub interface: Option<String>,
    #[serde(rename = "@allow-null")]
    pub allow_null: Option<bool>,
    #[serde(rename = "@enum")]
    pub r#enum: Option<String>,
}

#[derive(Debug, Deserialize)]
pub enum WaylandMessageType {
    #[serde(rename = "destructor")]
    Destructor,
}

#[derive(Debug, Deserialize)]
pub struct WaylandMessage {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@type")]
    pub r#type: Option<WaylandMessageType>,
    #[serde(rename = "@since")]
    pub since: Option<u32>,
    #[serde(rename = "description")]
    pub description: Option<WaylandDescription>,
    #[serde(rename = "arg")]
    pub args: Option<Vec<WaylandArg>>,
}

#[derive(Debug, Deserialize)]
pub struct WaylandEnumEntry {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@value")]
    pub value: String,
    #[serde(rename = "@summary")]
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct WaylandEnum {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@since")]
    pub since: Option<u32>,
    #[serde(rename = "@bitfield")]
    pub bitfield: Option<bool>,
    #[serde(rename = "description")]
    pub description: Option<WaylandDescription>,
    #[serde(rename = "entry")]
    pub entries: Vec<WaylandEnumEntry>,
}

#[derive(Debug, Deserialize)]
pub enum WaylandInterfaceItem {
    #[serde(rename = "request")]
    Request(WaylandMessage),
    #[serde(rename = "event")]
    Event(WaylandMessage),
    #[serde(rename = "enum")]
    Enum(WaylandEnum),
}

#[derive(Debug, Deserialize)]
pub struct WaylandInterface {
    #[serde(rename = "@version")]
    pub version: u32,
    #[serde(rename = "@name")]
    pub name: String,
    pub description: Option<WaylandDescription>,
    #[serde(rename = "$value")]
    pub items: Option<Vec<WaylandInterfaceItem>>,
}

#[derive(Debug, Deserialize)]
pub struct WaylandProtocol {
    #[serde(rename = "@name")]
    pub name: String,
    pub copyright: Option<String>,
    #[serde(rename = "interface")]
    pub interfaces: Vec<WaylandInterface>,
}

impl WaylandProtocol {
    pub fn load_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        Ok(quick_xml::de::from_reader(BufReader::new(File::open(path)?))?)
    }
}

struct ResolvedArg {
    data_type: WaylandDataType,
    interface: Option<InterfaceId>,
    allow_null: bool,
}

struct ResolvedMessage {
    since: u32,
    args: Vec<ResolvedArg>,
    message: WaylandMessage,
}

struct ResolvedInterface {
    version: u32,
    requests: Vec<ResolvedMessage>,
    events: Vec<ResolvedMessage>,
}

struct WaylandProtocolMapInner {
    interfaces: Vec<ResolvedInterface>,
    interface_lookup: BTreeMap<String, InterfaceId>,
}

fn add_message(collection: &mut Vec<ResolvedMessage>, msg: WaylandMessage) {
    let mut args = vec![];
    for arg in msg.args.as_ref().unwrap_or(&vec![]) {
        args.push(ResolvedArg {
            data_type: arg.data_type,
            interface: None,
            allow_null: arg.allow_null.unwrap_or(false),
        });
    }
    collection.push(ResolvedMessage { since: msg.since.unwrap_or(0), args, message: msg });
}

pub struct WaylandProtocolMapBuilder(WaylandProtocolMapInner);

impl WaylandProtocolMapBuilder {
    pub fn new() -> Self {
        Self(WaylandProtocolMapInner { interfaces: vec![], interface_lookup: BTreeMap::new() })
    }

    pub fn dir<P: AsRef<Path>>(self, path: P) -> anyhow::Result<Self> {
        let mut me = self;
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if entry.file_type()?.is_file() && entry.file_name().to_string_lossy().ends_with(".xml")
            {
                me = me.file(entry.path())?;
            }
        }
        Ok(me)
    }

    pub fn file<P: AsRef<Path>>(mut self, path: P) -> anyhow::Result<Self> {
        let protocol = WaylandProtocol::load_path(path)?;
        for interface in protocol.interfaces {
            if self.0.interface_lookup.contains_key(&interface.name) {
                bail!("duplicate interface {:?}", interface);
            }
            let mut requests = vec![];
            let mut events = vec![];
            if let (Some(items), name) = (interface.items, interface.name) {
                for item in items {
                    match item {
                        WaylandInterfaceItem::Request(m) => add_message(&mut requests, m),
                        WaylandInterfaceItem::Event(m) => add_message(&mut events, m),
                        _ => (),
                    }
                }
                self.0.interface_lookup.insert(name, InterfaceId(self.0.interfaces.len() as u16));
                self.0.interfaces.push(ResolvedInterface {
                    version: interface.version,
                    requests,
                    events,
                });
            }
        }
        Ok(self)
    }

    pub fn build(mut self) -> anyhow::Result<WaylandProtocolMap> {
        let WaylandProtocolMapInner { ref interface_lookup, ref mut interfaces } = self.0;
        for iface in interfaces {
            for msg in iface.requests.iter_mut().chain(iface.events.iter_mut()) {
                let empty_vec = vec![];
                let unresolved_args = msg.message.args.as_ref().unwrap_or(&empty_vec);
                for (arg, unresolved_arg) in msg.args.iter_mut().zip(unresolved_args) {
                    if let Some(interface_name) = unresolved_arg.interface.as_ref() {
                        if let Some(interface_id) = interface_lookup.get(interface_name) {
                            arg.interface = Some(*interface_id);
                        } else {
                            bail!("unknown interface {:?}", interface_name);
                        }
                    }
                }
            }
        }
        Ok(WaylandProtocolMap(Arc::new(self.0)))
    }
}

#[derive(Clone)]
pub struct WaylandProtocolMap(Arc<WaylandProtocolMapInner>);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct InterfaceId(u16);

enum LocalHandle {
    RequestHandler(InterfaceId),
    EventHandler(InterfaceId),
    Empty,
}

pub struct WaylandConnection {
    stream: UnixScmStream,
    protocol_map: WaylandProtocolMap,
    local_handle_table: Vec<LocalHandle>,
    local_free_list: Vec<usize>,
    cmsg_buf: Vec<u8>,
    rx_buf_max: usize,
    rx_buf_fd_max: usize,
    rx_buf: Vec<u8>,
    rx_buf_fill: usize,
    rx_fd_buf: Vec<Option<OwnedFd>>,
}

pub struct IncomingMessage<'a> {
    sender: u32,
    opcode: u16,
    message_spec: &'a ResolvedMessage,
    data: &'a [u8],
    fds: &'a mut Vec<Option<OwnedFd>>,
}

impl<'a> IncomingMessage<'a> {
    pub fn sender(&self) -> u32 {
        self.sender
    }

    pub fn opcode(&self) -> u16 {
        self.opcode
    }

    pub fn args<'b>(&'b mut self) -> IncomingMessageArgs<'b, 'a> {
        IncomingMessageArgs { message: self, buf_pos: 0, fd_pos: 0, n: 0 }
    }
}

pub enum IncomingMessageValue<'a> {
    Int(i32),
    Uint32(u32),
    Fixed, // TODO
    String(&'a str),
    Object(Option<u32>),
    NewId(InterfaceId, u32),
    Array(&'a [u8]),
    Fd(&'a mut Option<OwnedFd>),
}

pub struct IncomingMessageArg {
    data_type: WaylandDataType,
    interface: Option<InterfaceId>,
    data_range: Range<usize>,
    fd_index: Option<usize>,
}

pub struct IncomingMessageArgs<'b, 'a: 'b> {
    message: &'b mut IncomingMessage<'a>,
    buf_pos: usize,
    fd_pos: usize,
    n: usize,
}

impl<'b, 'a> IncomingMessageArgs<'b, 'a> {
    pub fn data(&self, arg: &IncomingMessageArg) -> &[u8] {
        &self.message.data[arg.data_range.clone()]
    }

    pub fn take_fd(&mut self, arg: &IncomingMessageArg) -> OwnedFd {
        let fd = self.message.fds[arg.fd_index.unwrap()].take().unwrap();
        fd
    }

    fn advance(&mut self) -> Option<IncomingMessageArg> {
        let arg_spec = match self.message.message_spec.args.get(self.n) {
            Some(arg) => arg,
            None => return None,
        };
        let length = match arg_spec.data_type {
            WaylandDataType::Int => 4,
            WaylandDataType::Uint => 4,
            WaylandDataType::Fixed => 4,
            WaylandDataType::String => {
                4 + round_up(
                    NativeEndian::read_u32(&self.message.data[self.buf_pos..self.buf_pos + 4])
                        as usize,
                    4,
                )
            }
            WaylandDataType::Object => 4,
            WaylandDataType::NewId => {
                if let Some(_interface) = arg_spec.interface.as_ref() {
                    4
                } else {
                    panic!("dynamic new_id not yet implemented");
                }
            }
            WaylandDataType::Array => {
                4 + round_up(
                    NativeEndian::read_u32(&self.message.data[self.buf_pos..self.buf_pos + 4])
                        as usize,
                    4,
                )
            }
            WaylandDataType::Fd => 4,
        };
        let has_fd = arg_spec.data_type == WaylandDataType::Fd;
        let data_range = self.buf_pos..self.buf_pos + length;
        let fd_index = if has_fd {
            let fd_pos = self.fd_pos;
            self.fd_pos += 1;
            Some(fd_pos)
        } else {
            None
        };
        self.buf_pos += length;
        self.n += 1;
        Some(IncomingMessageArg {
            data_type: arg_spec.data_type,
            interface: None,
            data_range,
            fd_index,
        })
    }
}

impl WaylandConnection {
    pub fn new(protocol_map: WaylandProtocolMap, stream: UnixScmStream) -> Self {
        let display_interface = protocol_map.0.interface_lookup.get("wl_display").unwrap();
        Self {
            stream,
            cmsg_buf: UnixScmStream::alloc_cmsg_buf(),
            local_handle_table: vec![
                LocalHandle::Empty,
                LocalHandle::RequestHandler(*display_interface),
            ],
            local_free_list: vec![],
            rx_buf_max: 65536,
            rx_buf_fd_max: 1024,
            rx_buf_fill: 0,
            rx_buf: vec![0; 65536],
            rx_fd_buf: vec![],
            protocol_map,
        }
    }

    pub fn message<'a>(&'a mut self) -> anyhow::Result<IncomingMessage<'a>> {
        if self.rx_buf_fill < 8 {
            bail!("no current message");
        }
        let sender = NativeEndian::read_u32(&self.rx_buf[0..4]);
        let opcode = NativeEndian::read_u16(&self.rx_buf[4..6]);
        let length = NativeEndian::read_u16(&self.rx_buf[6..8]) as usize;
        if self.rx_buf_fill < length {
            bail!("no current message");
        }
        let resolved_message = match self.local_handle_table.get(sender as usize) {
            Some(LocalHandle::RequestHandler(interface_id)) => {
                let interface = &self.protocol_map.0.interfaces[interface_id.0 as usize];
                let message = &interface.requests[opcode as usize];
                message
            }
            Some(LocalHandle::EventHandler(interface_id)) => {
                let interface = &self.protocol_map.0.interfaces[interface_id.0 as usize];
                let message = &interface.events[opcode as usize];
                message
            }
            _ => bail!("invalid message sender"),
        };
        Ok(IncomingMessage {
            sender,
            opcode,
            data: &self.rx_buf[8..length],
            fds: &mut self.rx_fd_buf,
            message_spec: resolved_message,
        })
    }

    pub async fn advance(&mut self) -> anyhow::Result<()> {
        if self.rx_buf_fill > 0 {
            self.consume_message()?;
        }
        self.fill_buffer(8, 0).await?;
        let msg_len = NativeEndian::read_u16(&self.rx_buf[6..8]) as usize;
        self.fill_buffer(msg_len, 0).await?;
        Ok(())
    }

    async fn fill_buffer(&mut self, bytes_needed: usize, fds_needed: usize) -> io::Result<()> {
        loop {
            if self.rx_buf_fill >= bytes_needed && self.rx_fd_buf.len() >= fds_needed {
                return Ok(());
            }
            if bytes_needed > self.rx_buf_max || fds_needed > self.rx_buf_fd_max {
                return Err(io::ErrorKind::InvalidInput.into());
            }
            let mut blocking = true;
            loop {
                let buf = &mut self.rx_buf[self.rx_buf_fill..];
                // TODO need an IoSliceMut version of this so we can do the deque thing
                let result =
                    self.stream.recv(buf, &mut self.cmsg_buf, &mut self.rx_fd_buf, blocking).await;
                match result {
                    Ok(nread) => {
                        println!("nread = {nread} bytes_needed = {bytes_needed}");
                        if nread == 0 {
                            // we may reach the end of the stream after reading
                            // enough data but before the buffer fill loop
                            // naturally exits.
                            if !blocking {
                                break;
                            }
                            return Err(io::ErrorKind::UnexpectedEof.into());
                        }
                        self.rx_buf_fill += nread;
                        if self.rx_fd_buf.len() > self.rx_buf_fd_max {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                "fd receive buffer overflow",
                            ));
                        }
                        blocking = false;
                    }
                    Err(e) => {
                        if e.kind() == io::ErrorKind::WouldBlock && !blocking {
                            break;
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
        }
    }

    fn consume_message(&mut self) -> anyhow::Result<()> {
        let mut msg = self.message()?;
        let mut consumed_fds = 0;
        let consumed_bytes = msg.data.len() + 8;
        let mut args = msg.args();
        while let Some(arg) = args.advance() {
            if arg.fd_index.is_some() {
                consumed_fds += 1;
            }
        }
        if consumed_bytes > self.rx_buf_fill || consumed_fds > self.rx_fd_buf.len() {
            panic!("message is larger than itself...?");
        }
        self.rx_fd_buf.drain(..consumed_fds);
        self.rx_buf.rotate_left(consumed_bytes);
        self.rx_buf_fill -= consumed_bytes;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use adrastea_core::net::UnixScmListener;
    use tempdir::TempDir;
    use tokio::net::UnixListener;
    use wayland_client::protocol::{wl_callback::WlCallback, wl_registry::WlRegistry};

    use super::*;

    #[test]
    pub fn load_all_protocols() -> anyhow::Result<()> {
        let proto_map = WaylandProtocolMapBuilder::new()
            .file("/home/eiz/code/wayland/protocol/wayland.xml")?
            .dir("/home/eiz/code/wayland-protocols/stable/xdg-shell")?
            .build()?;
        assert_eq!(proto_map.0.interfaces.len(), 27);
        Ok(())
    }

    #[tokio::test]
    pub async fn wayland_client_connect() -> anyhow::Result<()> {
        struct TestState;
        impl Dispatch<WlRegistry, ()> for TestState {
            fn event(
                _state: &mut Self, _proxy: &WlRegistry,
                event: <WlRegistry as wayland_client::Proxy>::Event, _data: &(),
                _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
            ) {
                println!("wl_registry {:?}", event);
            }
        }
        impl Dispatch<WlCallback, ()> for TestState {
            fn event(
                _state: &mut Self, _proxy: &WlCallback,
                event: <WlCallback as wayland_client::Proxy>::Event, _data: &(),
                _conn: &Connection, _qhandle: &wayland_client::QueueHandle<Self>,
            ) {
                println!("wl_callback {:?}", event);
            }
        }

        let test_dir = TempDir::new("wayland_client_connect")?;
        let sockpath = test_dir.path().join("wayland.sock");
        let listener = UnixScmListener::new(UnixListener::bind(&sockpath)?);
        async fn listener_task(mut listener: UnixScmListener) -> anyhow::Result<()> {
            let proto_map = WaylandProtocolMapBuilder::new()
                .file("/home/eiz/code/wayland/protocol/wayland.xml")?
                .build()?;
            let conn = listener.accept().await?;
            let mut conn = WaylandConnection::new(proto_map, conn);
            conn.advance().await?;
            let msg = conn.message()?;
            assert_eq!(msg.sender(), 1);
            assert_eq!(msg.opcode(), 1);
            println!("received get_registry");
            conn.advance().await?;
            let msg = conn.message()?;
            assert_eq!(msg.sender(), 1);
            assert_eq!(msg.opcode(), 0);
            println!("received sync");
            Ok(())
        }
        let jh = tokio::spawn(async move {
            listener_task(listener).await.unwrap();
        });
        tokio::task::spawn_blocking({
            let sockpath = sockpath.to_path_buf();
            move || {
                use std::os::unix::net::UnixStream;
                let sock = UnixStream::connect(&sockpath).unwrap();
                let conn = Connection::from_socket(sock).unwrap();
                let display = conn.display();
                let event_queue: EventQueue<TestState> = conn.new_event_queue();
                let handle = event_queue.handle();
                let _registry = display.get_registry(&handle, ());
                display.sync(&handle, ());
                event_queue.flush().unwrap();
            }
        })
        .await?;
        jh.await?;
        Ok(())
    }
}
