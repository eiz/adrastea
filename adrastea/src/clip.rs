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

// Portions of this file are derived from the CLIP project.
//
// Copyright (c) 2021 OpenAI
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use core::ops::Range;
use std::{fs::File, io::Write, path::Path, time::Instant};

use alloc::{
    collections::{BTreeMap, BinaryHeap},
    sync::Arc,
};
use anyhow::bail;
use bstr::{BString, ByteSlice};
use half::f16;
use memmap2::Mmap;
use regex::Regex;
use serde::Deserialize;
use simt_hip::{HipDevice, HipPhysicalDevice};
use skia_safe::{
    AlphaType, ColorSpace, ColorType, CubicResampler, Data, EncodedImageFormat, Image, ImageInfo,
    Pixmap, Surface,
};

use crate::{
    kernels::{
        BinaryOp, CommonKernels, GpuKernels, MatmulMask, MatmulOptions, MatmulStore, UnaryOp,
    },
    pickle::{load_tensor, PickledModel},
    tensor::{Tensor, TensorLayout, TensorView, TensorViewMut},
};

// this implements the 🤗 version of CLIP
// specifically clip-vit-large-patch14 atm

fn utf8_len(chr: u8) -> usize {
    const LOOKUP: &[usize] = &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(chr >> 4) as usize]
}

#[derive(Debug, Clone)]
struct Symbol {
    prev: i32,
    next: i32,
    text: Range<usize>,
}

#[derive(Debug, Eq, PartialEq)]
struct Bigram {
    left: i32,
    right: i32,
    score: usize,
    len: usize,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        std::cmp::Ordering::Equal
            .then(other.score.cmp(&self.score))
            .then(self.left.cmp(&other.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct ClipVocabulary {
    token_lookup: BTreeMap<BString, usize>,
    tokens: Vec<String>,
    word_regex: Arc<Regex>,
}

impl ClipVocabulary {
    pub fn new() -> Self {
        const VOCAB_BYTES: &[u8] = include_bytes!("../../assets/clip_vocab.json");
        let tokens: Vec<String> =
            serde_json::from_slice(VOCAB_BYTES).expect("malformed internal vocabulary");
        let mut token_lookup = BTreeMap::new();
        for string in &tokens {
            token_lookup.insert(string.as_bytes().into(), token_lookup.len());
        }
        Self {
            token_lookup,
            tokens,
            word_regex: Arc::new(Regex::new(
                r#"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#,
            ).unwrap()),
        }
    }
}

// CLIP's tokenizer is a bit ... odd ... and this code does not attempt to
// faithfully reproduce the exact result of the original Python code, which
// among other things depends on the 'ftfy' unicode hack-fix library and html
// unescaping.
//
// This is adapted from the tokenizer code I wrote for llama.cpp. Which itself
// is very similar to the original SentencePiece BPE tokenization algorithm. It
// should produce similar'ish output to the original clip code
struct ClipTokenizer {
    vocab: Arc<ClipVocabulary>,
    symbols: Vec<Symbol>,
    work_queue: BinaryHeap<Bigram>,
}

impl ClipTokenizer {
    pub fn new(vocab: Arc<ClipVocabulary>) -> Self {
        Self { vocab, symbols: Vec::new(), work_queue: BinaryHeap::new() }
    }

    pub fn encode(&mut self, text: &str) -> anyhow::Result<Vec<i32>> {
        let mut result = vec![];
        let word_regex = self.vocab.word_regex.clone();
        for rmatch in word_regex.find_iter(&text.to_lowercase()) {
            self.tokenize_word(&mut result, &format!("{}</w>", rmatch.as_str()))?;
        }
        Ok(result)
    }

    pub fn decode(&self, tokens: &[i32]) -> String {
        let mut result = String::new();
        let mut first = true;
        for token in tokens {
            result.push_str(&self.vocab.tokens[*token as usize].replace("</w>", " "));
        }
        if result.ends_with(' ') {
            result.pop();
        }
        result
    }

    fn tokenize_word(&mut self, result: &mut Vec<i32>, word: &str) -> anyhow::Result<()> {
        // split the string into utf8 characters
        let word = word.as_bytes();
        let mut index = 0;
        let mut offs = 0;
        while offs < word.len() {
            let mut char_len = utf8_len(word[offs]).min(word.len() - offs);
            if offs + char_len == word.len() - 4 {
                // special case: the magic </w> is part of the last symbol
                char_len += 4;
            }
            let sym = Symbol {
                prev: index - 1,
                next: if offs + char_len == word.len() { -1 } else { index + 1 },
                text: offs..offs + char_len,
            };
            offs += char_len;
            index += 1;
            self.symbols.push(sym);
        }
        // initial bigrams
        for i in 1..self.symbols.len() as i32 {
            self.try_add_bigram(word, i - 1, i);
        }
        // keep performing the highest priority substitution for as long as we can
        while let Some(bigram) = self.work_queue.pop() {
            let mut left_sym = self.symbols[bigram.left as usize].clone();
            let mut right_sym = self.symbols[bigram.right as usize].clone();
            // one of the symbols was already merged, skip
            if left_sym.text.len() == 0
                || right_sym.text.len() == 0
                || left_sym.text.len() + right_sym.text.len() != bigram.len
            {
                continue;
            }
            // merge the symbols
            left_sym.text = left_sym.text.start..right_sym.text.end;
            right_sym.text = 0..0;
            left_sym.next = right_sym.next;
            if right_sym.next >= 0 {
                self.symbols[right_sym.next as usize].prev = bigram.left;
            }
            self.symbols[bigram.left as usize] = left_sym.clone();
            self.symbols[bigram.right as usize] = right_sym.clone();
            // search for more bigrams
            self.try_add_bigram(word, left_sym.prev, bigram.left);
            self.try_add_bigram(word, bigram.left, left_sym.next);
        }
        let mut i = 0;
        while i != -1 {
            let symbol = &self.symbols[i as usize];
            let text = word[symbol.text.clone()].as_bstr();
            if let Some(token) = self.vocab.token_lookup.get(text) {
                result.push(*token as i32);
            } else {
                // TODO: I still don't fully get how bytes are supposed to work in this tokenizer
                bail!("failed to encode {:?}", text);
            }
            i = symbol.next;
        }
        self.symbols.clear();
        Ok(())
    }

    fn try_add_bigram(&mut self, word: &[u8], left: i32, right: i32) {
        if left == -1 || right == -1 {
            return;
        }
        let text =
            &word[self.symbols[left as usize].text.start..self.symbols[right as usize].text.end];
        if let Some(token_idx) = self.vocab.token_lookup.get(text.as_bstr()) {
            self.work_queue.push(Bigram { left, right, score: *token_idx, len: text.len() })
        }
    }
}

fn to_f16(kernels: &dyn CommonKernels, tensor: Tensor<f32>) -> anyhow::Result<Tensor<f16>> {
    let mut output = Tensor::new_hip_layout(tensor.layout().clone())?;
    kernels.fp32_to_fp16(&mut output.as_view_mut(), &tensor.as_view())?;
    Ok(output)
}

struct ClipModelLoader<'a> {
    pickle: &'a PickledModel<()>,
    kernels: &'a dyn CommonKernels,
    params: &'a ClipParams,
}

impl<'a> ClipModelLoader<'a> {
    pub fn new(
        pickle: &'a PickledModel<()>, kernels: &'a dyn CommonKernels, params: &'a ClipParams,
    ) -> Self {
        Self { pickle, kernels, params }
    }

    pub fn load_tensor_f16(&self, name: &str) -> anyhow::Result<Tensor<f16>> {
        to_f16(self.kernels, load_tensor(self.pickle, name)?)
    }
}

#[derive(Debug, Deserialize, Clone)]
struct ClipVisionParams {
    image_size: i32,
    patch_size: i32,
    num_attention_heads: i32,
    num_hidden_layers: i32,
}

#[derive(Debug, Deserialize, Clone)]
struct ClipTextParams {
    num_attention_heads: i32,
    num_hidden_layers: i32,
    vocab_size: i32,
}

#[derive(Debug, Deserialize, Clone)]
struct ClipParams {
    projection_dim: i32,
    vision_config: ClipVisionParams,
    text_config: ClipTextParams,
}

#[derive(Debug)]
struct ClipLayerNorm {
    weight: Tensor<f16>,
    bias: Tensor<f16>,
}

impl ClipLayerNorm {
    pub fn new(builder: &ClipModelLoader, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: builder.load_tensor_f16(&format!("{}.weight", prefix))?,
            bias: builder.load_tensor_f16(&format!("{}.bias", prefix))?,
        })
    }
}

struct ClipAttention {
    query: ClipLinear,
    key: ClipLinear,
    value: ClipLinear,
    out: ClipLinear,
}

impl ClipAttention {
    pub fn new(builder: &ClipModelLoader, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            query: ClipLinear::new(builder, &format!("{}.q_proj", prefix))?,
            key: ClipLinear::new(builder, &format!("{}.k_proj", prefix))?,
            value: ClipLinear::new(builder, &format!("{}.v_proj", prefix))?,
            out: ClipLinear::new(builder, &format!("{}.out_proj", prefix))?,
        })
    }
}

struct ClipLinear {
    weight: Tensor<f16>,
    bias: Tensor<f16>,
}

impl ClipLinear {
    pub fn new(builder: &ClipModelLoader, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: builder.load_tensor_f16(&format!("{}.weight", prefix))?,
            bias: builder.load_tensor_f16(&format!("{}.bias", prefix))?,
        })
    }
}

struct ClipMLP {
    fc1: ClipLinear,
    fc2: ClipLinear,
}

impl ClipMLP {
    pub fn new(builder: &ClipModelLoader, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            fc1: ClipLinear::new(builder, &format!("{}.fc1", prefix))?,
            fc2: ClipLinear::new(builder, &format!("{}.fc2", prefix))?,
        })
    }
}

struct ClipTransformerBlock {
    layer_norm1: ClipLayerNorm,
    attn: ClipAttention,
    layer_norm2: ClipLayerNorm,
    mlp: ClipMLP,
}

impl ClipTransformerBlock {
    pub fn new(builder: &ClipModelLoader, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            layer_norm1: ClipLayerNorm::new(builder, &format!("{}.layer_norm1", prefix))?,
            attn: ClipAttention::new(builder, &format!("{}.self_attn", prefix))?,
            layer_norm2: ClipLayerNorm::new(builder, &format!("{}.layer_norm2", prefix))?,
            mlp: ClipMLP::new(builder, &format!("{}.mlp", prefix))?,
        })
    }
}

struct ClipVisionTransformer {
    params: ClipVisionParams,
    class_embedding: Tensor<f16>,
    patch_embedding: Tensor<f16>,
    pre_layernorm: ClipLayerNorm,
    position_embedding: Tensor<f16>,
    post_layernorm: ClipLayerNorm,
    layers: Vec<ClipTransformerBlock>,
}

impl ClipVisionTransformer {
    fn new(builder: &ClipModelLoader, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            class_embedding: builder
                .load_tensor_f16(&format!("{}.embeddings.class_embedding", prefix))?,
            patch_embedding: builder
                .load_tensor_f16(&format!("{}.embeddings.patch_embedding.weight", prefix))?,
            position_embedding: builder
                .load_tensor_f16(&format!("{}.embeddings.position_embedding.weight", prefix))?,
            // [sic]
            pre_layernorm: ClipLayerNorm::new(builder, &format!("{}.pre_layrnorm", prefix))?,
            post_layernorm: ClipLayerNorm::new(builder, &format!("{}.post_layernorm", prefix))?,
            layers: (0..builder.params.vision_config.num_hidden_layers)
                .map(|i| {
                    ClipTransformerBlock::new(builder, &format!("{}.encoder.layers.{}", prefix, i))
                })
                .collect::<anyhow::Result<_>>()?,
            params: builder.params.vision_config.clone(),
        })
    }
}

struct ClipTextTransformer {
    position_embedding: Tensor<f16>,
    token_embedding: Tensor<f32>,
    layers: Vec<ClipTransformerBlock>,
    //
}

fn load_image_eager<P: AsRef<Path>>(path: P) -> anyhow::Result<Image> {
    let file = File::open(path)?;
    let map = unsafe { Mmap::map(&file)? };
    let data = unsafe { Data::new_bytes(&map) };
    Ok(Image::from_encoded(data)
        .ok_or_else(|| anyhow::anyhow!("failed to load image"))?
        .to_raster_image(None)
        .ok_or_else(|| anyhow::anyhow!("failed to load image"))?)
}

fn resize_dims(src_height: i32, src_width: i32, target_size: i32) -> (i32, i32) {
    if src_height <= src_width {
        (target_size, (target_size as f32 * src_width as f32 / src_height as f32) as i32)
    } else {
        ((target_size as f32 * src_height as f32 / src_width as f32) as i32, target_size)
    }
}

fn pixmap_as_planes(pixmap: Pixmap, channel_mean: &[f32], channel_stddev: &[f32]) -> Tensor<f32> {
    let dims = pixmap.dimensions();
    let (w, h) = (dims.width as usize, dims.height as usize);
    let mut planes = vec![0.0f32; h * w * 3];
    match pixmap.color_type() {
        // god's one correct pixel format. the goat
        skia_safe::ColorType::BGRA8888 => {
            for (i, pixel) in pixmap.pixels::<[u8; 4]>().unwrap().iter().enumerate() {
                planes[i] = ((pixel[2] as f32 / 255.0) - channel_mean[0]) / channel_stddev[0];
                planes[(w * h) + i] =
                    ((pixel[1] as f32 / 255.0) - channel_mean[1]) / channel_stddev[1];
                planes[(w * h * 2) + i] =
                    ((pixel[0] as f32 / 255.0) - channel_mean[2]) / channel_stddev[2];
            }
        }
        _ => todo!("unsupported color type"),
    }
    Tensor::from_vec(planes, TensorLayout::row_major(&[3, h, w]))
}

struct ClipImage(Tensor<f16>);

trait LazyClipImage {
    fn load(&self, kernels: &dyn CommonKernels) -> anyhow::Result<ClipImage>;
}

impl<P: AsRef<Path>> LazyClipImage for P {
    // this should likely be factored such that it just loads the image and puts
    // it into the right representation (planar float) and then the resize and
    // normalization is done in the model
    fn load(&self, kernels: &dyn CommonKernels) -> anyhow::Result<ClipImage> {
        let image = load_image_eager(self.as_ref())?;
        let pixdims = image.dimensions();
        let (new_height, new_width) = resize_dims(pixdims.height, pixdims.width, 224);
        let mut surface = Surface::new_raster(
            &ImageInfo::new(
                (224, 224),
                ColorType::BGRA8888,
                AlphaType::Premul,
                ColorSpace::new_srgb(),
            ),
            0,
            None,
        )
        .unwrap();
        let canvas = surface.canvas();
        canvas.translate((-((new_width - 224) / 2), -((new_height - 224) / 2)));
        canvas.scale((
            new_width as f32 / pixdims.width as f32,
            new_height as f32 / pixdims.height as f32,
        ));
        // TODO: this does not give the same results as PIL =/
        canvas.draw_image_with_sampling_options(image, (0, 0), CubicResampler::catmull_rom(), None);
        let pixmap = surface.peek_pixels().unwrap();
        let values = to_f16(
            kernels,
            pixmap_as_planes(
                pixmap,
                &[0.48145466, 0.4578275, 0.40821073],
                &[0.26862954, 0.26130258, 0.27577711],
            )
            .into_hip()?,
        )?;
        println!("{:>7.4?}", values);
        let snap = surface.image_snapshot();
        let context = surface.direct_context();
        let data = snap.encode(context, EncodedImageFormat::PNG, None).unwrap();
        let mut f = File::create("/home/eiz/clip_crop.png")?;
        f.write_all(data.as_bytes()).unwrap();
        Ok(ClipImage(values))
    }
}

struct ClipVisionContext {
    model: Arc<ClipVisionTransformer>,
    kernels: Arc<dyn CommonKernels>,
}

impl ClipVisionContext {
    pub fn new(model: Arc<ClipVisionTransformer>, kernels: Arc<dyn CommonKernels>) -> Self {
        Self { model, kernels }
    }

    pub fn encode(&self, image: impl LazyClipImage) -> anyhow::Result<Tensor<f16>> {
        let image = image.load(&*self.kernels)?;
        let mut embedding = Tensor::new_hip(&[257, 1024])?;
        let mut class_embed = embedding.as_view_mut();
        let mut class_embed = class_embed.take(&[1, 1024]);
        self.kernels.elementwise_unary_2d_f16(
            &mut class_embed,
            &self.model.class_embedding.as_view().shape_cast(&[1, 1024]),
            UnaryOp::Identity,
        )?;
        let mut patch_embeds = embedding.as_view_mut();
        let mut patch_embeds =
            patch_embeds.skip(&[1, 0]).shape_cast(&[16, 16, 1024]).permute(&[2, 0, 1]);
        let zero_bias = Tensor::new_hip(&[1024])?;
        self.kernels.conv2d_f16(
            &mut patch_embeds,
            &image.0.as_view(),
            &self.model.patch_embedding.as_view(),
            &zero_bias.as_view(),
            (14, 14),
            (0, 0),
            crate::kernels::Conv1dActivation::None,
        )?;
        self.kernels.elementwise_binary_2d_f16_inplace(
            &mut embedding.as_view_mut(),
            &self.model.position_embedding.as_view(),
            BinaryOp::Add,
        )?;
        let mut hidden_state = Tensor::new_hip(&[257, 1024])?;
        self.kernels.layer_norm(
            &mut hidden_state.as_view_mut(),
            &embedding.as_view(),
            &self.model.pre_layernorm.weight.as_view(),
            &self.model.pre_layernorm.bias.as_view(),
            1.0e-5,
        )?;
        for layer in &self.model.layers {
            self.process_layer(&mut hidden_state.as_view_mut(), layer, MatmulMask::None)?;
        }
        let mut output_norm = Tensor::new_hip(&[1, 1024])?;
        self.kernels.layer_norm(
            &mut output_norm.as_view_mut(),
            &hidden_state.as_view().take(&[1, 1024]),
            &self.model.post_layernorm.weight.as_view(),
            &self.model.post_layernorm.bias.as_view(),
            1.0e-5,
        )?;
        Ok(output_norm)
    }

    fn process_layer(
        &self, hidden_state: &mut TensorViewMut<f16>, layer: &ClipTransformerBlock,
        mask: MatmulMask,
    ) -> anyhow::Result<()> {
        let mut normed = Tensor::new_hip(&hidden_state.layout().dims)?;
        self.kernels.layer_norm(
            &mut normed.as_view_mut(),
            &hidden_state.as_view(),
            &layer.layer_norm1.weight.as_view(),
            &layer.layer_norm1.bias.as_view(),
            1.0e-5,
        )?;
        self.residual_attention(hidden_state, &normed.as_view(), &layer.attn, mask)?;
        self.kernels.layer_norm(
            &mut normed.as_view_mut(),
            &hidden_state.as_view(),
            &layer.layer_norm2.weight.as_view(),
            &layer.layer_norm2.bias.as_view(),
            1.0e-5,
        )?;
        self.residual_mlp(hidden_state, &normed.as_view(), &layer.mlp)?;
        Ok(())
    }

    fn residual_attention(
        &self, hidden_state: &mut TensorViewMut<f16>, normed: &TensorView<f16>,
        attn: &ClipAttention, mask: MatmulMask,
    ) -> anyhow::Result<()> {
        let heads = self.model.params.num_attention_heads as isize;
        let mut query = Tensor::new_hip(&normed.layout().dims)?;
        let mut key = Tensor::new_hip(&normed.layout().dims)?;
        let mut value = Tensor::new_hip(&normed.layout().dims)?;
        let mut qkv = Tensor::new_hip(&normed.layout().dims)?;
        self.kernels.matmul_f16(
            &mut query.as_view_mut(),
            normed,
            &attn.query.weight.as_view().permute(&[1, 0]),
            MatmulOptions::new().store(MatmulStore::BetaBias(0.0, &attn.query.bias.as_view())),
        )?;
        self.kernels.matmul_f16(
            &mut key.as_view_mut(),
            normed,
            &attn.key.weight.as_view().permute(&[1, 0]),
            MatmulOptions::new().store(MatmulStore::BetaBias(0.0, &attn.key.bias.as_view())),
        )?;
        self.kernels.matmul_f16(
            &mut value.as_view_mut(),
            normed,
            &attn.value.weight.as_view().permute(&[1, 0]),
            MatmulOptions::new().store(MatmulStore::BetaBias(0.0, &attn.value.bias.as_view())),
        )?;
        let q_view =
            query.as_view().shape_cast(&[query.size(-2) as isize, heads, -1]).permute(&[1, 0, 2]);
        let k_view =
            key.as_view().shape_cast(&[key.size(-2) as isize, heads, -1]).permute(&[1, 2, 0]);
        let v_view =
            value.as_view().shape_cast(&[value.size(-2) as isize, heads, -1]).permute(&[1, 0, 2]);
        let mut qk = Tensor::new_hip(&[heads as usize, q_view.size(-2), k_view.size(-1)])?;
        self.kernels.matmul_f16(
            &mut qk.as_view_mut(),
            &q_view,
            &k_view,
            MatmulOptions::new()
                .store(MatmulStore::Scale(
                    1.0 / (hidden_state.size(-1) as f32 / heads as f32).sqrt(),
                ))
                .mask(mask),
        )?;
        self.kernels.softmax_rows_inplace(&mut qk.as_view_mut(), 1.0)?;
        let mut qkv_view =
            qkv.as_view_mut().shape_cast(&[query.size(-2) as isize, heads, -1]).permute(&[1, 0, 2]);
        self.kernels.matmul_f16(&mut qkv_view, &qk.as_view(), &v_view, MatmulOptions::new())?;
        self.kernels.matmul_f16(
            hidden_state,
            &qkv.as_view(),
            &attn.out.weight.as_view().permute(&[1, 0]),
            MatmulOptions::new().store(MatmulStore::BetaBias(1.0, &attn.out.bias.as_view())),
        )?;
        Ok(())
    }

    fn residual_mlp(
        &self, hidden_state: &mut TensorViewMut<f16>, normed: &TensorView<f16>, mlp: &ClipMLP,
    ) -> anyhow::Result<()> {
        let mut mlp_hidden = Tensor::new_hip(&[hidden_state.size(-2), mlp.fc1.weight.size(-2)])?;
        self.kernels.matmul_f16(
            &mut mlp_hidden.as_view_mut(),
            normed,
            &mlp.fc1.weight.as_view().permute(&[1, 0]),
            MatmulOptions::new().store(MatmulStore::QuickGeluBias(&mlp.fc1.bias.as_view())),
        )?;
        self.kernels.matmul_f16(
            hidden_state,
            &mlp_hidden.as_view(),
            &mlp.fc2.weight.as_view().permute(&[1, 0]),
            MatmulOptions::new().store(MatmulStore::BetaBias(1.0, &mlp.fc2.bias.as_view())),
        )?;
        Ok(())
    }
}

pub fn clip_test<P: AsRef<Path>>(path: P) -> anyhow::Result<()> {
    let path = path.as_ref();
    let phys = HipPhysicalDevice::get(0)?;
    let device = Arc::new(HipDevice::new(phys)?);
    let _scope = device.lock()?;
    let kernels = Arc::new(GpuKernels::new(phys.capability()?)?);
    let model = PickledModel::load_file(path.join("pytorch_model.bin"), None)?;
    let params: ClipParams = serde_json::from_reader(File::open(path.join("config.json"))?)?;
    let builder = ClipModelLoader::new(&model, &*kernels, &params);
    let vt = ClipVisionTransformer::new(&builder, "vision_model")?;
    let context = ClipVisionContext::new(Arc::new(vt), kernels);
    println!("{:#?}", params);
    let mut tokenizer = ClipTokenizer::new(Arc::new(ClipVocabulary::new()));
    let tokens = tokenizer.encode("hello world thumbwar")?;
    println!("{:?}", tokens);
    println!("{:?}", tokenizer.decode(&tokens));
    context.encode("/home/eiz/clip_gold.png")?;
    todo!();
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use simt_hip::{HipDevice, HipPhysicalDevice};

    use crate::{
        kernels::{CommonKernels, GpuKernels, UnaryOp},
        pickle::{load_tensor, PickledModel},
        tensor::Tensor,
    };

    #[test]
    fn resize_dims_works() {
        assert_eq!(super::resize_dims(480, 640, 224), (224, 298));
        assert_eq!(super::resize_dims(640, 480, 224), (298, 224));
    }

    #[test]
    fn conv2d_matches_pytorch() -> anyhow::Result<()> {
        let phys = HipPhysicalDevice::get(0)?;
        let device = Arc::new(HipDevice::new(phys)?);
        let _scope = device.lock()?;
        let kernels = Arc::new(GpuKernels::new(phys.capability()?)?);
        let model = PickledModel::load_file("/tmp/clip_tests.pt", None)?;
        let pixel_values: Tensor<f32> = load_tensor(&model, "clip.conv2d.pixel_values")?;
        let weight: Tensor<f32> = load_tensor(&model, "clip.conv2d.weight")?;
        let class_embed: Tensor<f32> = load_tensor(&model, "clip.class_embed")?;
        let expected_class_patch_embed: Tensor<f32> =
            load_tensor(&model, "clip.class_patch_embed")?;
        let mut embedding = Tensor::new_hip(&[257, 1024])?;
        let mut class_embed_dest = embedding.as_view_mut();
        let mut class_embed_dest = class_embed_dest.take(&[1, 1024]);
        kernels.elementwise_unary_2d_f32(
            &mut class_embed_dest,
            &class_embed.as_view().shape_cast(&[1, 1024]),
            UnaryOp::Identity,
        )?;
        let mut actual_result = embedding.as_view_mut();
        let mut actual_result =
            actual_result.skip(&[1, 0]).shape_cast(&[16, 16, 1024]).permute(&[2, 0, 1]);
        let mut err_stats_gpu = Tensor::new_hip(&[3])?;
        let bias = Tensor::new_hip(&[1024])?;
        kernels.conv2d_f32(
            &mut actual_result,
            &pixel_values.as_view(),
            &weight.as_view(),
            &bias.as_view(),
            (14, 14),
            (0, 0),
            crate::kernels::Conv1dActivation::None,
        )?;
        kernels.error_stats_f32(
            &mut err_stats_gpu.as_view_mut(),
            &embedding.as_view(),
            &expected_class_patch_embed.as_view(),
        )?;
        let err_stats = err_stats_gpu.into_cpu()?;
        let err_stats = err_stats.storage().as_cpu();
        println!("expected result {:>7.4?}", expected_class_patch_embed);
        println!("actual result {:>7.4?}", embedding);
        println!("err_stats {:?}", err_stats);
        assert!(err_stats[0] < 0.001);
        Ok(())
    }
}
