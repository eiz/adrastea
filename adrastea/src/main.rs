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
use adrastea_core::{
    net::{UnixScmListener, UnixScmStream},
    util::{AtomicRing, AtomicRingReader, AtomicRingWriter, IUnknown},
};
use adrastea_media::{
    audio::{AudioControlThread, NUM_CHANNELS, SAMPLE_RATE},
    vulkan,
    wayland::{ISkiaPaint, SurfaceClient},
    wayland_protocol::{
        MessageReaderValue, WaylandConnection, WaylandConnectionRole, WaylandProtocolMap,
        WaylandProtocolMapBuilder, WaylandReceiver, WaylandSender,
    },
};
use adrastea_models::{
    clip,
    kernels::{CommonKernels, GpuKernels, MatmulOptions, MatmulTracer},
    llama::{LlamaContext, LlamaModel, LlamaParams, MetaLlamaModelLoader},
    llava,
    pickle::{ModelState, PickledModel},
    tensor::Tensor,
    whisper::{
        WhisperContext, WhisperModel, WhisperModelState, WHISPER_CHUNK_LENGTH, WHISPER_SAMPLE_RATE,
    },
};
use alloc::{collections::VecDeque, sync::Arc};
use core::{
    any::Provider,
    cell::RefCell,
    fmt::{Debug, Display, Formatter},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use mathpix::{ImageToTextOptions, MathPixClient};
use rlua::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    ffi::OsStr,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};
use tokio::net::UnixListener;
use walkdir::WalkDir;

use anyhow::bail;
use atspi::{
    connection::AccessibilityConnection,
    proxy::{accessible::AccessibleProxy, cache::CacheProxy},
};
use clap::{Parser, Subcommand};
use half::f16;
use sentencepiece::SentencePieceProcessor;
use simt::{Gpu, GpuModule, Kernel, LaunchParams, PhysicalGpu};
use skia_safe::{paint::Style, Canvas, Color, Font, FontStyle, Paint, Typeface};

extern crate alloc;

pub mod mathpix;

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
    let phys = PhysicalGpu::any().expect("no gpu found");
    let device = Arc::new(Gpu::new(phys)?);
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
    let phys = PhysicalGpu::any().expect("no gpu found");
    let device = Arc::new(Gpu::new(phys)?);
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
    let phys = PhysicalGpu::any().expect("no gpu found");
    let device = Arc::new(Gpu::new(phys)?);
    let _scope = device.lock()?;
    let kernels = GpuKernels::new(phys.capability()?)?;
    let module_microbench = GpuModule::find(phys.capability()?, adrastea_kernels::microbench)?;
    let empty_kernel: Kernel<(i32,)> = Kernel::new(&module_microbench, "empty_kernel")?;
    let wmma_loop_f16_f16: Result<Kernel<(i32,)>, simt::Error> =
        Kernel::new(&module_microbench, "wmma_loop_f16_f16");
    let wmma_loop_f32_f16: Result<Kernel<(i32,)>, simt::Error> =
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

    let left = Tensor::new_gpu(&[2048, 4096])?;
    let right = Tensor::new_gpu(&[4096, 4096])?;
    let mut out = Tensor::new_gpu(&[2048, 4096])?;
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

fn llama_test<P: AsRef<Path>>(path: P) -> anyhow::Result<()> {
    let path = path.as_ref();
    let phys = PhysicalGpu::any().expect("no gpu found");
    let device = Arc::new(Gpu::new(phys)?);
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

#[derive(Parser)]
struct Opt {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    Load {
        #[arg(value_name = "FILE")]
        path: PathBuf,
        #[arg(long)]
        dict_path: Option<String>,
    },
    LoadWhisper {
        #[arg(value_name = "FILE")]
        path: PathBuf,
    },
    Wav {
        #[arg(value_name = "WAV")]
        wav_path: PathBuf,
        #[arg(value_name = "MODEL")]
        model_path: PathBuf,
    },
    Vulkan,
    Microbenchmark,
    Audio,
    Wayland,
    Combined,
    Llama {
        #[arg(value_name = "MODEL")]
        path: PathBuf,
    },
    Clip {
        #[arg(value_name = "MODEL")]
        path: PathBuf,
    },
    Llava {
        #[arg(value_name = "LLAVA")]
        llava_path: PathBuf,
        #[arg(value_name = "CLIP")]
        clip_path: PathBuf,
        #[arg(value_name = "IMAGES")]
        images: Vec<PathBuf>,
        #[arg(
            long,
            short,
            help = "The text prompt. $0, $1, $2, ... $9 are replaced with images.",
            default_value = "Describe the image.$1"
        )]
        prompt: String,
    },
    Atk,
    Mpix,
    Wserver {
        #[arg(value_name = "LISTEN")]
        listen_path: PathBuf,
        #[arg(value_name = "SERVER")]
        server_path: PathBuf,
    },
}

#[async_recursion::async_recursion]
async fn print_ax_tree(
    ax: &AccessibilityConnection, root: &AccessibleProxy<'_>, level: usize,
) -> anyhow::Result<()> {
    let role = root.get_role().await?;
    if level == 0 {
        println!("{}", role);
    } else {
        println!("{}{} {:?}", " ".repeat(level * 2), root.get_role().await?, root.name().await?);
    }
    if role == atspi::Role::Application && false {
        let cache = CacheProxy::builder(ax.connection())
            .destination(root.destination())?
            .path("/org/a11y/atspi/cache")?
            .build()
            .await?;
        let items = cache.get_items().await?;
        for item in items {
            println!("{}{:?}", " ".repeat((level + 1) * 2), item);
        }
    } else {
        let children = root.get_children().await?;
        for (source, path) in children {
            let child = AccessibleProxy::builder(ax.connection())
                .destination(source)?
                .path(path)?
                .build()
                .await?;
            print_ax_tree(ax, &child, level + 1).await?;
        }
    }
    Ok(())
}

#[tokio::main]
async fn atk_test() -> anyhow::Result<()> {
    atspi::connection::set_session_accessibility(true).await?;
    let ax = AccessibilityConnection::open().await?;
    let desktop = AccessibleProxy::builder(ax.connection())
        .destination("org.a11y.atspi.Registry")?
        .path("/org/a11y/atspi/accessible/root")?
        .build()
        .await?;
    print_ax_tree(&ax, &desktop, 0).await?;
    Ok(())
}

#[tokio::main]
async fn mpix_test() -> anyhow::Result<()> {
    let config = MpixConfig::load_from_file("/home/eiz/.mathpix")?;
    let mut img_bytes = vec![];
    std::io::stdin().read_to_end(&mut img_bytes)?;
    let client = MathPixClient::new(&config.app_id, &config.app_key);
    let options = ImageToTextOptions::default();
    let response = client.image_to_text(img_bytes, options).await?;
    println!("response {:#?}", response);
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct MpixConfig {
    app_id: String,
    app_key: String,
}

impl MpixConfig {
    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        Ok(serde_yaml::from_reader(file)?)
    }
}

pub struct LuaWaylandProxy {
    protocol_map: WaylandProtocolMap,
    server_path: PathBuf,
    listener: UnixScmListener,
    lua: Lua,
}

impl LuaWaylandProxy {
    pub fn bind(
        protocol_map: WaylandProtocolMap, path: impl AsRef<Path>, server_path: impl Into<PathBuf>,
    ) -> anyhow::Result<Self> {
        let lua = Lua::new();
        lua.context(|c| -> anyhow::Result<()> {
            let src = std::fs::read_to_string("/home/eiz/adrastea/proxy.lua")?;
            c.load(&src).exec()?;
            Ok(())
        })?;
        let listener = UnixScmListener::new(UnixListener::bind(path)?);
        Ok(Self { protocol_map, server_path: server_path.into(), listener, lua })
    }

    pub async fn listen(&self) -> anyhow::Result<()> {
        loop {
            let stream = self.listener.accept().await?;
            let server_path = self.server_path.clone();
            let protocol_map = self.protocol_map.clone();
            tokio::spawn(async move {
                if let Err(e) = deno_wp_main(protocol_map, server_path, stream).await {
                    eprintln!("wayland proxy connection exited with error: {:?}", e);
                }
            });
        }
    }
}

async fn deno_wp_forward(
    mut rx: WaylandReceiver, mut tx: WaylandSender, name: &str,
) -> anyhow::Result<()> {
    loop {
        rx.advance().await?;
        let mut msg = rx.message()?;
        println!("{} forwarding {}", name, msg.debug_name());
        let mut builder = tx.message_builder(msg.sender(), msg.opcode())?;
        let mut args = msg.args();
        while let Some(arg) = args.advance() {
            println!("  arg: {:?}", args.value(&arg)?);
            match args.value(&arg)? {
                MessageReaderValue::Int(value) => builder = builder.int(value),
                MessageReaderValue::Uint(value) => builder = builder.uint(value),
                MessageReaderValue::Fixed(value) => builder = builder.fixed(value),
                MessageReaderValue::String(value) => builder = builder.string(value),
                MessageReaderValue::Object(value) => builder = builder.object(value),
                MessageReaderValue::NewId(interface_id, object_id, version) => {
                    builder = builder.new_id(interface_id, object_id, version)
                }
                MessageReaderValue::Array(value) => builder = builder.array(value),
                MessageReaderValue::Fd(_) => {
                    // TODO we can get rid of fd_owned by having like borrow_fd on args or something
                    builder = builder.fd_owned(args.take_fd(&arg));
                }
            }
        }
        builder.send().await?;
    }
}

async fn deno_wp_main(
    protocol_map: WaylandProtocolMap, server_path: PathBuf, stream: UnixScmStream,
) -> anyhow::Result<()> {
    let (server_rx, server_tx) =
        WaylandConnection::new(protocol_map.clone(), stream, WaylandConnectionRole::Server);
    let client_stream = UnixScmStream::connect(server_path).await?;
    let (client_rx, client_tx) =
        WaylandConnection::new(protocol_map, client_stream, WaylandConnectionRole::Client);
    // TODO deal with annoying tokio e wrapping stuff (flatten out results) for non-panicking try_join
    let server_jh = tokio::spawn(async move {
        if let Err(e) = deno_wp_forward(server_rx, client_tx, "client->server").await {
            eprintln!("wayland proxy connection exited with error: {:?}", e);
        }
    });
    let client_jh = tokio::spawn(async move {
        if let Err(e) = deno_wp_forward(client_rx, server_tx, "server->client").await {
            eprintln!("wayland proxy connection exited with error: {:?}", e);
        }
    });
    tokio::try_join!(server_jh, client_jh)?;
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn wserver_test(listen_path: &Path, server_path: &Path) -> anyhow::Result<()> {
    let mut proto_builder = WaylandProtocolMapBuilder::new()
        .file("/home/eiz/code/wayland/protocol/wayland.xml")?
        .dir("/home/eiz/code/wayland-protocols/staging/fractional-scale")?
        .dir("/home/eiz/code/wayland-protocols/staging/xdg-activation")?
        .dir("/home/eiz/code/wayland-protocols/unstable/xdg-output")?
        .dir("/home/eiz/code/wayland-protocols/unstable/primary-selection")?
        .dir("/home/eiz/code/wayland-protocols/unstable/text-input")?;
    for entry in WalkDir::new("/home/eiz/code/wayland-protocols/stable") {
        let entry = entry?;
        if entry.path().extension() == Some(OsStr::new("xml")) {
            proto_builder = proto_builder.file(entry.path())?;
        }
    }
    let proto_map = proto_builder.build()?;
    let proxy = LuaWaylandProxy::bind(proto_map, listen_path, server_path)?;
    proxy.listen().await?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::parse();
    match opt.command {
        CliCommand::Load { path, dict_path } => {
            let model = PickledModel::load_file(&path, dict_path.as_ref().map(|s| s.as_str()))?;
            println!("{:#?}", model.tensors);
        }
        CliCommand::LoadWhisper { path } => {
            let model = PickledModel::load_typed::<WhisperModelState, _>(&path, ())?;
            println!("{:#?}", model.tensors);
            println!("{:#?}", model.metadata);
        }
        CliCommand::Wav { wav_path, model_path } => {
            wav_test(wav_path, model_path)?;
        }
        CliCommand::Vulkan => unsafe { vulkan::vulkan_square()? },
        CliCommand::Microbenchmark => unsafe { microbenchmark()? },
        CliCommand::Audio => streaming_test(None)?,
        CliCommand::Wayland => {
            let (_audio_state, surface_state) = gui_test_state();
            wayland_test(surface_state)?
        }
        CliCommand::Combined => {
            let (audio_state, surface_state) = gui_test_state();
            std::thread::spawn(move || streaming_test(Some(audio_state)));
            wayland_test(surface_state)?;
        }
        CliCommand::Llama { path } => llama_test(&path)?,
        CliCommand::Clip { path } => clip::clip_test(&path)?,
        CliCommand::Llava { llava_path, clip_path, images, prompt } => {
            llava::llava_test(llava_path, clip_path, &images, &prompt)?
        }
        CliCommand::Atk => atk_test()?,
        CliCommand::Mpix => mpix_test()?,
        CliCommand::Wserver { listen_path, server_path } => {
            wserver_test(&listen_path, &server_path)?
        }
    }
    Ok(())
}
