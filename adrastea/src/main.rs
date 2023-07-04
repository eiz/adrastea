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

#![feature(provide_any, iter_advance_by)]
use alloc::{collections::VecDeque, sync::Arc};
use core::{
    any::Provider,
    cell::RefCell,
    ffi::{c_void, CStr},
    fmt::{Debug, Display, Formatter},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use llama::MetaLlamaModelLoader;
use std::{collections::HashMap, fs::File, io::Write, path::Path, time::Instant};

use anyhow::bail;
use half::f16;
use sentencepiece::SentencePieceProcessor;
use serde::{Deserialize, Serialize};
use simt_hip::{HipDevice, HipModule, HipPhysicalDevice, Kernel, LaunchParams};
use skia_safe::{
    paint::Style, Canvas, Color, EncodedImageFormat, Font, FontStyle, Paint, Surface, Typeface,
};
use wayland::{ISkiaPaint, SurfaceClient};

use crate::{
    audio::{AudioControlThread, NUM_CHANNELS, SAMPLE_RATE},
    kernels::{CommonKernels, GpuKernels, MatmulOptions, MatmulTracer},
    llama::{LlamaContext, LlamaModel, LlamaParams},
    pickle::{ModelState, PickledModel},
    tensor::Tensor,
    util::{AtomicRing, AtomicRingReader, AtomicRingWriter, IUnknown},
    whisper::{
        WhisperContext, WhisperModel, WhisperModelState, WHISPER_CHUNK_LENGTH, WHISPER_SAMPLE_RATE,
    },
};

extern crate alloc;

pub mod audio;
pub mod clip;
pub mod kernels;
pub mod llama;
pub mod llava;
pub mod mel;
pub mod pickle;
pub mod rt_alloc;
pub mod stft;
pub mod tensor;
pub mod util;
pub mod vulkan;
pub mod wayland;
pub mod whisper;

#[inline(always)]
unsafe fn cuda_call<F: FnOnce() -> simt_cuda_sys::CUresult>(
    cb: F,
) -> Result<(), simt_cuda_sys::CUresult> {
    let res = cb();
    if res == simt_cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(res)
    }
}

#[inline(always)]
unsafe fn cuda_result_call<T, F: FnOnce(*mut T) -> simt_cuda_sys::CUresult>(
    cb: F,
) -> Result<T, simt_cuda_sys::CUresult> {
    let mut out = std::mem::MaybeUninit::uninit();
    let res = cb(out.as_mut_ptr());
    if res == simt_cuda_sys::CUresult::CUDA_SUCCESS {
        Ok(out.assume_init())
    } else {
        Err(res)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TritonKernelMetadata {
    num_warps: u32,
    num_stages: u32,
    constants: HashMap<String, u32>,
    debug: bool,
    shared: u32,
    name: String,
}

struct CudaContext {
    cuda: Arc<simt_cuda_sys::cuda>,
    device: simt_cuda_sys::CUdevice,
    context: simt_cuda_sys::CUcontext,
}

impl CudaContext {
    pub unsafe fn new(cuda: Arc<simt_cuda_sys::cuda>, device_index: i32) -> anyhow::Result<Self> {
        let device = cuda_result_call(|x| cuda.cuDeviceGet(x, device_index))?;
        let context = cuda_result_call(|x| cuda.cuCtxCreate_v2(x, 0, device))?;
        cuda_result_call(|x| cuda.cuCtxPopCurrent_v2(x)).expect("cuCtxPopCurrent_v2 failed");
        Ok(Self { cuda, device, context })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            cuda_call(|| self.cuda.cuCtxDestroy_v2(self.context)).expect("cuCtxDestroy_v2 failed");
        }
    }
}

// exercise for the reader: make get() not have to return a clone
thread_local! {
    static THREAD_CUDA_CONTEXT: RefCell<Option<Arc<CudaContext>>> = RefCell::new(None);
}

struct ScopedCudaContext {
    old_value: Option<Arc<CudaContext>>,
}

impl ScopedCudaContext {
    unsafe fn new(value: Arc<CudaContext>) -> Result<Self, simt_cuda_sys::CUresult> {
        let old_value = THREAD_CUDA_CONTEXT.with(|v| {
            let mut v = v.borrow_mut();
            let old_value = v.clone();
            cuda_call(|| value.cuda.cuCtxPushCurrent_v2(value.context))?;
            *v = Some(value);
            Ok(old_value)
        })?;
        Ok(ScopedCudaContext { old_value })
    }

    fn get() -> Result<Arc<CudaContext>, simt_cuda_sys::CUresult> {
        THREAD_CUDA_CONTEXT
            .with(|v| v.borrow().clone())
            .ok_or(simt_cuda_sys::CUresult::CUDA_ERROR_INVALID_CONTEXT)
    }

    pub fn capability() -> Result<i32, simt_cuda_sys::CUresult> {
        unsafe {
            let ctx = ScopedCudaContext::get()?;
            let major = cuda_result_call(|x| {
                ctx.cuda.cuDeviceGetAttribute(
                    x,
                    simt_cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    ctx.device,
                )
            })?;
            let minor = cuda_result_call(|x| {
                ctx.cuda.cuDeviceGetAttribute(
                    x,
                    simt_cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    ctx.device,
                )
            })?;
            Ok(major * 10 + minor)
        }
    }
}

impl Drop for ScopedCudaContext {
    fn drop(&mut self) {
        THREAD_CUDA_CONTEXT.with(|v| {
            let mut v = v.borrow_mut();
            unsafe {
                cuda_result_call(|x| v.as_ref().unwrap().cuda.cuCtxPopCurrent_v2(x))
                    .expect("cuCtxPopCurrent_v2 failed");
            }
            *v = self.old_value.clone()
        });
    }
}

struct CudaBuffer {
    ptr: simt_cuda_sys::CUdeviceptr,
    size: usize,
}

impl CudaBuffer {
    pub unsafe fn new(size: usize) -> anyhow::Result<Self> {
        let ctx = ScopedCudaContext::get()?;
        let ptr = cuda_result_call(|x| ctx.cuda.cuMemAlloc_v2(x, size))?;
        Ok(Self { ptr, size })
    }

    pub unsafe fn copy_from(
        &mut self, src: *const std::ffi::c_void, size: usize,
    ) -> anyhow::Result<()> {
        let ctx = ScopedCudaContext::get()?;
        cuda_call(|| ctx.cuda.cuMemcpyHtoD_v2(self.ptr, src, size))?;
        Ok(())
    }

    pub unsafe fn copy_to(&self, dst: *mut std::ffi::c_void, size: usize) -> anyhow::Result<()> {
        let ctx = ScopedCudaContext::get()?;
        cuda_call(|| ctx.cuda.cuMemcpyDtoH_v2(dst, self.ptr, size))?;
        Ok(())
    }

    pub fn copy_from_slice<T: Copy>(&mut self, src: &[T]) -> anyhow::Result<()> {
        assert_eq!(src.len() * std::mem::size_of::<T>(), self.size);
        unsafe { self.copy_from(src.as_ptr() as *const std::ffi::c_void, self.size) }
    }

    pub fn copy_to_slice<T: Copy>(&self, dst: &mut [T]) -> anyhow::Result<()> {
        assert_eq!(dst.len() * std::mem::size_of::<T>(), self.size);
        unsafe { self.copy_to(dst.as_mut_ptr() as *mut std::ffi::c_void, self.size) }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            let ctx = ScopedCudaContext::get()
                .expect("invariant: CudaBuffer must be dropped in a context scope");
            cuda_call(|| ctx.cuda.cuMemFree_v2(self.ptr)).expect("cuMemFree_v2 failed");
        }
    }
}

pub struct CudaModule {
    inner: simt_cuda_sys::CUmodule,
}

impl CudaModule {
    pub unsafe fn new(data: &[u8]) -> anyhow::Result<Self> {
        let ctx = ScopedCudaContext::get()?;
        let inner = cuda_result_call(|x| ctx.cuda.cuModuleLoadData(x, data.as_ptr() as *const _))?;
        Ok(Self { inner })
    }

    pub unsafe fn find(capability: i32, kernels: &[(&str, &[u8])]) -> anyhow::Result<Self> {
        let ctx = ScopedCudaContext::get()?;
        let mut compatible_kernels = vec![];
        for (arch, bin) in kernels {
            if !arch.starts_with("sm_") {
                continue;
            }
            let arch = arch[3..].parse::<i32>()?;
            if arch <= capability {
                compatible_kernels.push((arch, bin));
            }
        }
        compatible_kernels.sort_by_key(|(arch, _)| *arch);
        let (_, bin) = compatible_kernels
            .iter()
            .rev()
            .filter(|(arch, _)| *arch <= capability)
            .last()
            .ok_or_else(|| anyhow::anyhow!("no compatible kernel found"))?;
        let inner = cuda_result_call(|x| ctx.cuda.cuModuleLoadData(x, bin.as_ptr() as *const _))?;
        Ok(Self { inner })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        unsafe {
            let ctx = ScopedCudaContext::get()
                .expect("invariant: CudaModule must be dropped in a context scope");
            cuda_call(|| ctx.cuda.cuModuleUnload(self.inner)).expect("cuModuleUnload failed");
        }
    }
}

const ROWS: u64 = 8;
const COLS: u64 = 8;

unsafe fn cuda_square() -> anyhow::Result<()> {
    #[cfg(target_os = "linux")]
    const LIB: &str = "libcuda.so";
    #[cfg(windows)]
    const LIB: &str = "nvcuda.dll";
    let cuda = Arc::new(simt_cuda_sys::cuda::new(LIB)?);
    cuda_call(|| cuda.cuInit(0))?;
    let device_count = cuda_result_call(|x| cuda.cuDeviceGetCount(x))?;
    println!("{} device(s)", device_count);
    for i in 0..device_count {
        let mut name = [0u8; 256];
        cuda_call(|| cuda.cuDeviceGetName(name.as_mut_ptr() as *mut _, 256, i))?;
        let c_name = CStr::from_ptr(name.as_ptr() as *const _);
        println!("Device {}: {}", i, c_name.to_str()?);
    }
    if device_count == 0 {
        bail!("can't continue, no devices");
    }
    let context = Arc::new(CudaContext::new(cuda.clone(), 0)?);
    let _scoped_ctx = ScopedCudaContext::new(context.clone());
    let capability = ScopedCudaContext::capability()?;
    let module = CudaModule::find(capability, adrastea_kernels::square_fp32_16x16)?;
    let kernel = cuda_result_call(|x| {
        cuda.cuModuleGetFunction(x, module.inner, b"square_fp32_16x16\0".as_ptr() as *const i8)
    })?;
    dbg!(kernel);
    let stream = cuda_result_call(|x| cuda.cuStreamCreate(x, 0))?;
    dbg!(stream);
    let mut stage_buf = vec![0.0f32; (COLS * ROWS) as usize];
    let buf_sz = (COLS * ROWS * std::mem::size_of::<f32>() as u64) as usize;
    for y in 0..ROWS {
        for x in 0..COLS {
            stage_buf[(y * COLS + x) as usize] = (y + x) as f32;
        }
    }
    let mut buf = CudaBuffer::new(buf_sz)?;
    buf.copy_from_slice(&stage_buf)?;
    let grid_x = util::ceil_div(COLS, 16);
    let grid_y = util::ceil_div(ROWS, 16);
    let width = COLS as u32;
    let height = ROWS as u32;
    cuda_call(|| {
        cuda.cuLaunchKernel(
            kernel,
            grid_x as u32,
            grid_y as u32,
            1,
            16,
            16,
            1,
            0,
            stream,
            &[
                &buf.ptr as *const _ as *mut c_void,
                &buf.ptr as *const _ as *mut c_void,
                &width as *const _ as *mut c_void,
                &height as *const _ as *mut c_void,
            ] as *const _ as *mut _,
            std::ptr::null_mut(),
        )
    })?;
    cuda_call(|| cuda.cuStreamSynchronize(stream))?;
    buf.copy_to_slice(&mut stage_buf)?;
    for y in 0..ROWS {
        for x in 0..COLS {
            print!("{:4} ", stage_buf[(y * COLS + x) as usize]);
        }
        println!("");
    }
    Ok(())
}

fn wav2float_mono(data: &wav::BitDepth) -> Vec<f32> {
    match data {
        wav::BitDepth::Eight(v) => v.iter().map(|x| *x as f32 / 128.0 - 1.0).collect(),
        wav::BitDepth::Sixteen(v) => v.iter().map(|x| *x as f32 / 32768.0).collect(),
        wav::BitDepth::TwentyFour(v) => v.iter().map(|x| *x as f32 / 8388608.0).collect(),
        wav::BitDepth::ThirtyTwoFloat(v) => v.iter().map(|x| *x as f32).collect(),
        wav::BitDepth::Empty => vec![],
    }
}

fn wav_test<P: AsRef<Path>, Q: AsRef<Path>>(path: P, model_path: Q) -> anyhow::Result<()> {
    let phys = HipPhysicalDevice::get(0)?;
    let device = Arc::new(HipDevice::new(phys)?);
    let _scope = device.lock()?;
    // BIG TODO: loading each kernel as a separate module like this is super not ergonomic
    // use a better way
    let kernels = Arc::new(MatmulTracer::new(GpuKernels::new(phys.capability()?)?));
    let start = Instant::now();
    let model = WhisperModel::new(&WhisperModelState::load(model_path, ())?)?;
    println!("model load time: {:?}", start.elapsed());
    let mut context = WhisperContext::new(Arc::new(model), kernels.clone())?;
    let mut fp = File::open(path)?;
    // TODO: 'wav' eager loads everything =/
    let (header, data) = wav::read(&mut fp)?;
    println!("{:#?}", header);
    println!("{:#?}", context.model().dims());
    let data_len = match &data {
        wav::BitDepth::Eight(v) => v.len(),
        wav::BitDepth::Sixteen(v) => v.len(),
        wav::BitDepth::TwentyFour(v) => v.len(),
        wav::BitDepth::ThirtyTwoFloat(v) => v.len(),
        wav::BitDepth::Empty => 0,
    };
    println!("samples: {}", data_len / header.channel_count as usize);
    println!(
        "duration: {}s",
        data_len as f32 / (header.sampling_rate as f32 * header.channel_count as f32)
    );
    if header.sampling_rate != WHISPER_SAMPLE_RATE {
        bail!(
            "unsupported sample rate {} x{}, resample to 16khz mono",
            header.sampling_rate,
            header.channel_count
        );
    }
    let mut wave = wav2float_mono(&data);
    wave.extend(std::iter::repeat(0.0).take(WHISPER_SAMPLE_RATE as usize * WHISPER_CHUNK_LENGTH));
    let wave = &wave[0..WHISPER_SAMPLE_RATE as usize * WHISPER_CHUNK_LENGTH];
    let start = Instant::now();
    let features = context.encode(wave)?;
    println!("encode time: {:?}", start.elapsed());
    let mut tokens = (context.model().tokenizer())
        // language of glorious mother nation
        .encode_with_special_tokens("<|startoftranscript|><|en|><|transcribe|>")
        .iter()
        .map(|x| *x as i32)
        .collect::<Vec<_>>();
    let end_of_text = context.model().tokenizer().encode_with_special_tokens("<|endoftext|>")[0];
    println!("initial tokens {:?}", tokens);
    println!("features {:>7.4?}", features);
    let start = Instant::now();
    let mut total_generated = 0;
    for _i in 0..context.model().dims().n_text_ctx {
        let logits = context.decode(features.as_view(), &tokens)?.into_cpu()?;
        let logits_vec = logits.storage().as_cpu();
        let last_logits = &logits_vec[logits_vec.len() - context.model().dims().n_vocab as usize..];
        let argmax = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        println!("token {:>7.4?}", argmax);
        tokens.push(argmax as i32);
        let detok =
            context.model().tokenizer().decode(tokens.iter().map(|x| *x as usize).collect());
        println!("text {:?}", detok);
        total_generated += 1;
        if argmax as usize == end_of_text {
            break;
        }
    }
    println!(
        "decode time: {:?} ({:.4}s/tok)",
        start.elapsed(),
        start.elapsed().as_secs_f32() / total_generated as f32
    );
    for shape in kernels.shapes() {
        println!("{:?}", shape);
    }
    Ok(())
}

pub struct GuiTestShared {
    vad_on: AtomicBool,
}

pub struct GuiTestAudioThreadState {
    shared: Arc<GuiTestShared>,
    writer: AtomicRingWriter<String>,
}

impl Debug for GuiTestAudioThreadState {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GuiTestAudioThreadState").finish_non_exhaustive()
    }
}

struct MainWindowInner {
    reader: AtomicRingReader<String>,
    frame_number: usize,
    last_render: Duration,
    last_msg: String,
    font: Font,
}
pub struct MainWindow {
    shared: Arc<GuiTestShared>,
    inner: RefCell<MainWindowInner>,
}
impl MainWindow {
    pub fn new(shared: Arc<GuiTestShared>, reader: AtomicRingReader<String>) -> Self {
        let typeface = Typeface::from_name("monospace", FontStyle::normal()).unwrap();
        let font = Font::from_typeface(typeface, 18.0);
        let inner = MainWindowInner {
            reader,
            frame_number: 0,
            last_render: Duration::ZERO,
            last_msg: String::new(),
            font,
        };
        Self { shared, inner: RefCell::new(inner) }
    }
}
impl Debug for MainWindow {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MainWindow").finish_non_exhaustive()
    }
}
impl IUnknown for MainWindow {}
impl Provider for MainWindow {
    fn provide<'a>(&'a self, demand: &mut std::any::Demand<'a>) {
        demand.provide_ref::<dyn ISkiaPaint>(self);
    }
}
impl ISkiaPaint for MainWindow {
    fn on_paint_skia(&self, canvas: &mut Canvas, width: f32, height: f32) {
        let start = Instant::now();
        let mut inner = self.inner.borrow_mut();
        inner.frame_number += 1;
        while let Some(text) = inner.reader.try_pop() {
            inner.last_msg = text;
        }
        let mut paint = Paint::default();
        let mut text_paint = Paint::default();
        paint
            .set_style(Style::Stroke)
            .set_stroke_width(4.0)
            .set_color(Color::RED)
            .set_anti_alias(true);
        text_paint.set_color(Color::from_rgb(0xc0, 0xc0, 0xc0));
        canvas.clear(Color::from_rgb(0x20, 0x20, 0x20));
        for i in 0..((((inner.frame_number % 120) as i32) - 60).abs() / 2) + 3 {
            canvas.draw_circle((width / 1.25, height / 1.25), 128.0 + 8.0 * i as f32, &paint);
        }
        let txt = format!(
            "{:60} frame {:7}\n[VOICE] {:?}\n> ",
            "[Speech Recognition / Language Model Test]", inner.frame_number, inner.last_msg,
        );
        let mut y = 24;
        for line in txt.split('\n') {
            canvas.draw_str(line, (0, y), &inner.font, &text_paint);
            y += 24;
        }
        for color in &[
            Color::RED,
            Color::GREEN,
            Color::CYAN,
            Color::YELLOW,
            Color::MAGENTA,
            Color::BLUE,
            Color::from_rgb(0x80, 0x80, 0xe0),
        ] {
            text_paint.set_color(*color);
            canvas.draw_str("ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789 `!@#$%^&*(){}[]~;',./<>?:\"_+", (0, y), &inner.font, &text_paint);
            y += 24;
        }
        text_paint.set_color(Color::from_rgb(0x80, 0x80, 0xe0));
        canvas.draw_str(format!("{:>15.4?}", inner.last_render), (0, y), &inner.font, &text_paint);
        if self.shared.vad_on.load(Ordering::Relaxed) {
            canvas.draw_str(format!("{:>60}", "VAD"), (0, 24), &inner.font, &text_paint);
        }
        inner.last_render = start.elapsed();
    }
}

fn gui_test_state() -> (GuiTestAudioThreadState, MainWindow) {
    let shared = Arc::new(GuiTestShared { vad_on: AtomicBool::new(false) });

    let (reader, writer) = AtomicRing::new(128);
    (GuiTestAudioThreadState { shared: shared.clone(), writer }, MainWindow::new(shared, reader))
}

pub fn wayland_test(test_state: MainWindow) -> anyhow::Result<()> {
    let (mut event_queue, mut client) = SurfaceClient::connect_to_env()?;
    client.create_toplevel_surface(test_state);
    loop {
        event_queue.blocking_dispatch(&mut client)?;
    }
}

pub fn streaming_test(mut test_state: Option<GuiTestAudioThreadState>) -> anyhow::Result<()> {
    let phys = HipPhysicalDevice::get(0)?;
    let device = Arc::new(HipDevice::new(phys)?);
    let _scope = device.lock()?;
    let kernels = Arc::new(GpuKernels::new(phys.capability()?)?);
    let start = Instant::now();
    let model =
        WhisperModel::new(&WhisperModelState::load("/home/eiz/.cache/whisper/small.pt", ())?)?;
    println!("model load time: {:?}", start.elapsed());
    let mut context = WhisperContext::new(Arc::new(model), kernels.clone())?;
    let mut all_samples = VecDeque::new();
    let initial_tokens = (context.model().tokenizer())
        // language of glorious mother nation
        .encode_with_special_tokens("<|startoftranscript|><|en|><|transcribe|>")
        .iter()
        .map(|x| *x as i32)
        .collect::<Vec<_>>();
    let mut token_buffer = initial_tokens.clone();
    let end_of_text = context.model().tokenizer().encode_with_special_tokens("<|endoftext|>")[0];
    let audio_control = AudioControlThread::new()?;
    let mut audio_stream = audio_control.capture_audio_stream(Duration::from_millis(100))?;
    let timer = Instant::now();
    let mut vad_active = false;
    let mut vad_grace = 0;
    let mut prev_samples = [0.0f32; SAMPLE_RATE as usize / 10 * NUM_CHANNELS];
    loop {
        let mut vad_was_active = vad_active;
        let mut samples = [0.0f32; SAMPLE_RATE as usize / 10 * NUM_CHANNELS];
        audio_stream.next(&mut samples);
        let mut sum_sq = 0.0;
        for sample in &samples {
            sum_sq += sample * sample;
        }
        let rms = (sum_sq / samples.len() as f32).sqrt();
        if rms > 0.05 {
            if !vad_was_active {
                println!("vad active");
            }
            vad_active = true;
            vad_was_active = true;
            vad_grace = 10;
        } else {
            if vad_grace > 0 {
                vad_grace -= 1;
            } else {
                vad_active = false;
            }
            prev_samples = samples;
        }
        if vad_was_active {
            if all_samples.len() == 0 {
                all_samples.extend(&prev_samples);
            }
            all_samples.extend(samples.iter());
            if all_samples.len() > SAMPLE_RATE as usize * NUM_CHANNELS * 30 {
                all_samples.drain(0..all_samples.len() - SAMPLE_RATE as usize * NUM_CHANNELS * 30);
            }
        }
        if let Some(state) = test_state.as_mut() {
            state.shared.vad_on.store(vad_active, Ordering::Relaxed);
        }
        if !vad_was_active || vad_active {
            continue;
        }
        all_samples.extend(
            std::iter::repeat(0.0)
                .take(WHISPER_SAMPLE_RATE as usize * WHISPER_CHUNK_LENGTH - all_samples.len()),
        );
        println!("[{:?}] encoding", timer.elapsed());
        let features = context.encode(all_samples.make_contiguous())?;
        println!("[{:?}] decoding", timer.elapsed());
        for _i in 0..(context.model().dims().n_text_ctx as usize - token_buffer.len()) {
            let logits = context.decode(features.as_view(), &token_buffer)?.into_cpu()?;
            let logits_vec = logits.storage().as_cpu();
            let last_logits =
                &logits_vec[logits_vec.len() - context.model().dims().n_vocab as usize..];
            let argmax = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            if argmax as usize == end_of_text {
                break;
            }
            token_buffer.push(argmax as i32);
        }
        let detok = context.model().tokenizer().decode(
            token_buffer.iter().map(|x| *x as usize).filter(|&x| x < end_of_text).collect(),
        )?;
        let detok = detok.trim();
        println!("[{:?}] final: {}", timer.elapsed(), detok);
        if let Some(state) = test_state.as_mut() {
            state.writer.try_push(detok.into());
        }
        token_buffer = initial_tokens.clone();
        all_samples.clear();
    }
}

struct Flops(f64);

impl Display for Flops {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut flops = self.0;
        let mut units = 0;
        while flops > 1000.0 && units < 5 {
            flops /= 1000.0;
            units += 1;
        }
        let unit = match units {
            0 => " op/s",
            1 => "Kop/s",
            2 => "Mop/s",
            3 => "Gop/s",
            4 => "Top/s",
            _ => "Pop/s",
        };
        write!(f, "{:>7.4} {}", flops, unit)
    }
}

fn bench<F: FnMut() -> anyhow::Result<usize>>(name: &str, mut f: F) -> anyhow::Result<()> {
    let mut runs = vec![];
    let ops = f()?; // warmup
    let test_start = Instant::now();
    while test_start.elapsed().as_secs_f32() < 5.0 && runs.len() < 100000 {
        let start = Instant::now();
        let ops = f()?;
        runs.push((ops, start.elapsed()));
    }
    let (avg_ops, avg_elapsed) =
        runs.iter().fold((0, Duration::from_secs(0)), |(acc_ops, acc_elapsed), (ops, elapsed)| {
            (acc_ops + ops, acc_elapsed + *elapsed)
        });
    let avg_elapsed = avg_elapsed.as_secs_f64() / runs.len() as f64;
    let avg_ops = avg_ops as f64 / runs.len() as f64;
    let (min_ops, min_elapsed) = runs.iter().fold(
        (std::usize::MAX, Duration::MAX),
        |(acc_ops, acc_elapsed), (ops, elapsed)| {
            if *elapsed < acc_elapsed {
                (*ops, *elapsed)
            } else {
                (acc_ops, acc_elapsed)
            }
        },
    );
    let (max_ops, max_elapsed) =
        runs.iter().fold((0, Duration::from_secs(0)), |(acc_ops, acc_elapsed), (ops, elapsed)| {
            if *elapsed > acc_elapsed {
                (*ops, *elapsed)
            } else {
                (acc_ops, acc_elapsed)
            }
        });
    let avg_per_sec = avg_ops / avg_elapsed;
    let min_per_sec = min_ops as f64 / min_elapsed.as_secs_f64();
    let max_per_sec = max_ops as f64 / max_elapsed.as_secs_f64();
    println!(
        "{:40} {:>15} {:>15} {:>15} {:>15}",
        name,
        format!("{}", ops),
        format!("{}", Flops(min_per_sec)),
        format!("{}", Flops(avg_per_sec)),
        format!("{}", Flops(max_per_sec))
    );
    Ok(())
}

fn sync() -> anyhow::Result<()> {
    unsafe {
        simt_hip_sys::library().hipDeviceSynchronize();
    }
    Ok(())
}

#[inline(always)]
pub unsafe fn rocblas_call<F: FnOnce() -> simt_rocblas_sys::rocblas_status>(
    cb: F,
) -> anyhow::Result<()> {
    let res = cb();
    if res == simt_rocblas_sys::rocblas_status::rocblas_status_success {
        Ok(())
    } else {
        bail!("rocblas error {:?}", res);
    }
}

#[inline(always)]
pub unsafe fn rocblas_result_call<T, F: FnOnce(*mut T) -> simt_rocblas_sys::rocblas_status>(
    cb: F,
) -> anyhow::Result<T> {
    let mut out = std::mem::MaybeUninit::uninit();
    let res = cb(out.as_mut_ptr());
    if res == simt_rocblas_sys::rocblas_status::rocblas_status_success {
        Ok(out.assume_init())
    } else {
        bail!("rocblas error {:?}", res);
    }
}

unsafe fn microbenchmark() -> anyhow::Result<()> {
    let phys = HipPhysicalDevice::get(0)?;
    let device = Arc::new(HipDevice::new(phys)?);
    let _scope = device.lock()?;
    let kernels = GpuKernels::new(phys.capability()?)?;
    let module_microbench = HipModule::find(phys.capability()?, adrastea_kernels::microbench)?;
    let empty_kernel: Kernel<(i32,)> = Kernel::new(&module_microbench, "empty_kernel")?;
    let wmma_loop_f16_f16: Result<Kernel<(i32,)>, simt_hip::Error> =
        Kernel::new(&module_microbench, "wmma_loop_f16_f16");
    let wmma_loop_f32_f16: Result<Kernel<(i32,)>, simt_hip::Error> =
        Kernel::new(&module_microbench, "wmma_loop_f32_f16");
    let wgp_count = unsafe {
        simt_hip::hip_result_call(|x| {
            simt_hip_sys::library().hipDeviceGetAttribute(
                x,
                simt_hip_sys::hipDeviceAttribute_t::hipDeviceAttributeMultiprocessorCount,
                phys.index(),
            )
        })? as u32
    };
    println!("WGPs: {}", wgp_count);
    println!("{:40} {:>15} {:>15} {:>15} {:>15}", "name", "ops", "fast", "avg", "slow");

    bench("empty_kernel", || {
        empty_kernel.launch(
            LaunchParams { blocks: (1, 1, 1), threads: (1, 1, 1), shared_mem: 0, stream: None },
            (0,),
        )?;
        sync()?;
        Ok(1)
    })?;

    let left = Tensor::new_hip(&[2048, 4096])?;
    let right = Tensor::new_hip(&[4096, 4096])?;
    let mut out = Tensor::new_hip(&[2048, 4096])?;
    if let Ok(wmma_loop) = wmma_loop_f16_f16 {
        bench("wmma_loop_f16_f16", || {
            wmma_loop.launch(
                LaunchParams {
                    blocks: (wgp_count, 1, 1),
                    threads: (32, 4, 1),
                    shared_mem: 0,
                    stream: None,
                },
                (10000,),
            )?;
            sync()?;
            Ok(2 * 16 * 16 * 16 * 10000 * wgp_count as usize * 4)
        })?;
    }

    if let Ok(wmma_loop) = wmma_loop_f32_f16 {
        bench("wmma_loop_f32_f16", || {
            wmma_loop.launch(
                LaunchParams {
                    blocks: (wgp_count, 1, 1),
                    threads: (32, 4, 1),
                    shared_mem: 0,
                    stream: None,
                },
                (10000,),
            )?;
            sync()?;
            Ok(2 * 16 * 16 * 16 * 10000 * wgp_count as usize * 4)
        })?;
    }

    bench("matmul_f16_2048_4096_4096_rrr", || {
        kernels.matmul_f16_slow(
            &mut out.as_view_mut(),
            &left.as_view(),
            &right.as_view(),
            MatmulOptions::new(),
        )?;
        sync()?;
        Ok(2 * 2048 * 4096 * 4096)
    })?;
    bench("matmul_f16_2048_4096_4096_rrc", || {
        kernels.matmul_f16_slow(
            &mut out.as_view_mut(),
            &left.as_view(),
            &right.as_view().permute(&[1, 0]),
            MatmulOptions::new(),
        )?;
        sync()?;
        Ok(2 * 2048 * 4096 * 4096)
    })?;
    bench("matmul_f16_fast_2048_4096_4096_rrr", || {
        kernels.matmul_f16(
            &mut out.as_view_mut(),
            &left.as_view(),
            &right.as_view(),
            MatmulOptions::new(),
        )?;
        sync()?;
        Ok(2 * 2048 * 4096 * 4096)
    })?;
    bench("matmul_f16_fast_2048_4096_4096_rrc", || {
        kernels.matmul_f16(
            &mut out.as_view_mut(),
            &left.as_view(),
            &right.as_view().permute(&[1, 0]),
            MatmulOptions::new(),
        )?;
        sync()?;
        Ok(2 * 2048 * 4096 * 4096)
    })?;

    // Warning: running these tests first instead of last made the entire GPU driver crash
    // on Linux 6.3.6-arch1-1 for me, 100% repro. Does not seem to happen this way. I'm
    // assuming some kind of state corruption is happening but not sure where yet.
    let blas = simt_rocblas_sys::rocblas::new("librocblas.so")?;
    let blas_handle = rocblas_result_call(|x| blas.rocblas_create_handle(x))?;
    let one = simt_rocblas_sys::rocblas_half { data: f16::from_f32(1.0).to_bits() };
    let zero = simt_rocblas_sys::rocblas_half { data: f16::from_f32(0.0).to_bits() };

    bench("hgemm_f16_2048_4096_4096", || {
        rocblas_call(|| {
            blas.rocblas_hgemm(
                blas_handle,
                simt_rocblas_sys::rocblas_operation::rocblas_operation_none,
                simt_rocblas_sys::rocblas_operation::rocblas_operation_none,
                2048,
                4096,
                4096,
                &one,
                left.as_gpu_ptr() as *const simt_rocblas_sys::rocblas_half,
                2048,
                right.as_gpu_ptr() as *const simt_rocblas_sys::rocblas_half,
                4096,
                &zero,
                out.as_mut_gpu_ptr() as *mut simt_rocblas_sys::rocblas_half,
                2048,
            )
        })?;
        sync()?;
        Ok(2 * 2048 * 4096 * 4096)
    })?;

    bench("hgemm_f16_2048_4096_4096_nt", || {
        rocblas_call(|| {
            blas.rocblas_hgemm(
                blas_handle,
                simt_rocblas_sys::rocblas_operation::rocblas_operation_none,
                simt_rocblas_sys::rocblas_operation::rocblas_operation_transpose,
                2048,
                4096,
                4096,
                &one,
                left.as_gpu_ptr() as *const simt_rocblas_sys::rocblas_half,
                2048,
                right.as_gpu_ptr() as *const simt_rocblas_sys::rocblas_half,
                4096,
                &zero,
                out.as_mut_gpu_ptr() as *mut simt_rocblas_sys::rocblas_half,
                2048,
            )
        })?;
        sync()?;
        Ok(2 * 2048 * 4096 * 4096)
    })?;
    Ok(())
}

fn skia_test() -> anyhow::Result<()> {
    let mut surface = Surface::new_raster_n32_premul((256, 256)).unwrap();
    let canvas = surface.canvas();
    let mut paint = Paint::default();

    paint.set_style(Style::Stroke).set_stroke_width(4.0).set_color(Color::RED);
    canvas.draw_line((0, 0), (256, 256), &paint);
    let image = surface.image_snapshot();
    let data = image.encode(surface.direct_context(), EncodedImageFormat::PNG, None).unwrap();
    let mut f = File::create("test.png")?;
    f.write_all(data.as_bytes())?;
    Ok(())
}

fn llama_test<P: AsRef<Path>>(path: P) -> anyhow::Result<()> {
    let path = path.as_ref();
    let phys = HipPhysicalDevice::get(0)?;
    let device = Arc::new(HipDevice::new(phys)?);
    let _scope = device.lock()?;
    // BIG TODO: loading each kernel as a separate module like this is super not ergonomic
    // use a better way
    let kernels = Arc::new(MatmulTracer::new(GpuKernels::new(phys.capability()?)?));

    // TODO: support the various sharded model formats.
    let model = PickledModel::load_file(path.join("consolidated.00.pth"), None)?;
    let params: LlamaParams = serde_json::from_reader(File::open(path.join("params.json"))?)?;
    let tokenizer = SentencePieceProcessor::open(path.join("tokenizer.model"))?;
    let end_of_text = 1;
    let mut context = LlamaContext::new(
        Arc::new(LlamaModel::new(&MetaLlamaModelLoader::new(model), params, tokenizer, 0)?),
        kernels,
    );
    let mut token_buffer = vec![context.model().tokenizer().bos_id().unwrap() as i32];
    for _i in 0..200 {
        let logits = context.decode(&token_buffer)?.into_cpu()?;
        let logits_vec = logits.storage().as_cpu();
        let last_logits =
            &logits_vec[logits_vec.len() - context.model().params().vocab_size as usize..];
        let argmax = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        if argmax as usize == end_of_text {
            break;
        }
        token_buffer.push(argmax as i32);
        println!(
            "text {:?}",
            context
                .model()
                .tokenizer()
                .decode_piece_ids(&token_buffer.iter().map(|x| *x as u32).collect::<Vec<_>>())
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    println!("The endless sea.");
    if args.len() >= 2 && args[1] == "cuda" {
        unsafe { cuda_square()? }
    } else if args.len() >= 3 && args[1] == "load" {
        let dict_path = if args.len() >= 4 { Some(args[3].as_str()) } else { None };
        let model = PickledModel::load_file(&args[2], dict_path)?;
        println!("{:#?}", model.tensors);
    } else if args.len() >= 3 && args[1] == "load_whisper" {
        let model = PickledModel::load_typed::<WhisperModelState, _>(&args[2], ())?;
        println!("{:#?}", model.tensors);
        println!("{:#?}", model.metadata);
    } else if args.len() >= 4 && args[1] == "wav" {
        wav_test(&args[2], &args[3])?;
    } else if args.len() >= 2 && args[1] == "vulkan" {
        unsafe { vulkan::vulkan_square()? }
    } else if args.len() >= 2 && args[1] == "microbenchmark" {
        unsafe { microbenchmark()? }
    } else if args.len() >= 2 && args[1] == "audio" {
        streaming_test(None)?
    } else if args.len() >= 2 && args[1] == "wayland" {
        let (_audio_state, surface_state) = gui_test_state();
        wayland_test(surface_state)?
    } else if args.len() >= 2 && args[1] == "skia" {
        skia_test()?
    } else if args.len() >= 2 && args[1] == "combined" {
        let (audio_state, surface_state) = gui_test_state();
        std::thread::spawn(move || streaming_test(Some(audio_state)));
        wayland_test(surface_state)?;
    } else if args.len() >= 3 && args[1] == "llama" {
        llama_test(&args[2])?
    } else if args.len() >= 3 && args[1] == "clip" {
        clip::clip_test(&args[2])?
    } else if args.len() >= 2 && args[1] == "llava" {
        llava::llava_test()?
    } else {
        println!("test commands: cuda, load, wav, vulkan, microbenchmark, audio, wayland, skia, combined, llama, clip");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        tensor::{Tensor, TensorLayout},
        util::ElidingRangeIterator,
    };

    #[test]
    fn test_print_tensor() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], TensorLayout::row_major(&[2, 2]));
        println!("{:?}", tensor);
        let tensor = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            TensorLayout::row_major(&[3, 3]),
        );
        println!("standard\n{:?}\n", tensor);
        println!("transpose\n{:?}", tensor.as_view().permute(&[1, 0]));
    }

    #[test]
    fn test_elided_range() {
        let mut indices = vec![];
        for (_skip, i) in ElidingRangeIterator::new(10, 6, 3) {
            indices.push(i);
        }
        assert_eq!(indices, vec![0, 1, 2, 7, 8, 9]);
    }
}
