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

use alloc::sync::Arc;
use core::fmt::{self, Debug, Formatter};

use half::f16;
use rustfft::num_complex::Complex32;
use serde::Deserialize;
use tiktoken_rs::CoreBPE;

use crate::{
    kernels::{
        BinaryOp, CommonKernels, Conv1dActivation, MatmulLoad, MatmulMask, MatmulOptions,
        MatmulStore,
    },
    mel,
    pickle::{self, load_tensor, PickledModel},
    tensor::{Tensor, TensorLayout, TensorStorage, TensorView, TensorViewMut},
};

pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_N_MELS: usize = 80;
pub const WHISPER_HOP_LENGTH: usize = 160;
pub const WHISPER_CHUNK_LENGTH: usize = 30;
pub const WHISPER_CHUNK_FRAMES: usize =
    WHISPER_CHUNK_LENGTH * WHISPER_SAMPLE_RATE as usize / WHISPER_HOP_LENGTH;

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct WhisperDims {
    pub n_mels: i32,
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
}

#[derive(Deserialize)]
pub struct WhisperModelState {
    dims: WhisperDims,
    model_state_dict: serde_pickle::Value,
}

impl<'de> pickle::ModelState<'de> for WhisperModelState {
    type Metadata = WhisperDims;
    type LoadParams = ();
    fn state_dict(&self) -> &serde_pickle::Value {
        &self.model_state_dict
    }
    fn into_metadata(self) -> Self::Metadata {
        self.dims
    }
}

#[derive(Debug)]
pub struct WhisperLayerNorm {
    weight: Tensor<f16>,
    bias: Tensor<f16>,
}

impl WhisperLayerNorm {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: load_tensor(pickle, &format!("{}.weight", prefix))?,
            bias: load_tensor(pickle, &format!("{}.bias", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperAttention {
    query: WhisperLinear,
    key: Tensor<f16>,
    value: WhisperLinear,
    out: WhisperLinear,
}

impl WhisperAttention {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            query: WhisperLinear::new(pickle, &format!("{}.query", prefix))?,
            key: load_tensor(pickle, &format!("{}.key.weight", prefix))?,
            value: WhisperLinear::new(pickle, &format!("{}.value", prefix))?,
            out: WhisperLinear::new(pickle, &format!("{}.out", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperLinear {
    weight: Tensor<f16>,
    bias: Tensor<f16>,
}

impl WhisperLinear {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: load_tensor(pickle, &format!("{}.weight", prefix))?,
            bias: load_tensor(pickle, &format!("{}.bias", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperTransformerBlock {
    attn: WhisperAttention,
    cross_attn: Option<WhisperAttention>,
    cross_attn_ln: Option<WhisperLayerNorm>,
    attn_ln: WhisperLayerNorm,
    mlp_0: WhisperLinear,
    mlp_2: WhisperLinear,
    mlp_ln: WhisperLayerNorm,
}

impl WhisperTransformerBlock {
    pub fn new(
        pickle: &PickledModel<WhisperDims>, prefix: &str, has_cross_attn: bool,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            attn: WhisperAttention::new(pickle, &format!("{}.attn", prefix))?,
            cross_attn: if has_cross_attn {
                Some(WhisperAttention::new(pickle, &format!("{}.cross_attn", prefix))?)
            } else {
                None
            },
            cross_attn_ln: if has_cross_attn {
                Some(WhisperLayerNorm::new(pickle, &format!("{}.cross_attn_ln", prefix))?)
            } else {
                None
            },
            attn_ln: WhisperLayerNorm::new(pickle, &format!("{}.attn_ln", prefix))?,
            mlp_0: WhisperLinear::new(pickle, &format!("{}.mlp.0", prefix))?,
            mlp_2: WhisperLinear::new(pickle, &format!("{}.mlp.2", prefix))?,
            mlp_ln: WhisperLayerNorm::new(pickle, &format!("{}.mlp_ln", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperConv1d {
    weight: Tensor<f16>,
    bias: Tensor<f16>,
}

impl WhisperConv1d {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: load_tensor(pickle, &format!("{}.weight", prefix))?,
            bias: load_tensor(pickle, &format!("{}.bias", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperAudioEncoder {
    conv1: WhisperConv1d,
    conv2: WhisperConv1d,
    position_embedding: Tensor<f16>,
    layers: Vec<WhisperTransformerBlock>,
    ln_post: WhisperLayerNorm,
}

impl WhisperAudioEncoder {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            conv1: WhisperConv1d::new(pickle, &format!("{}.conv1", prefix))?,
            conv2: WhisperConv1d::new(pickle, &format!("{}.conv2", prefix))?,
            position_embedding: sinusoid_position_embedding(&pickle.metadata).into_hip()?,
            layers: (0..pickle.metadata.n_audio_layer)
                .map(|i| {
                    WhisperTransformerBlock::new(pickle, &format!("{}.blocks.{}", prefix, i), false)
                })
                .collect::<anyhow::Result<Vec<_>>>()?,
            ln_post: WhisperLayerNorm::new(pickle, &format!("{}.ln_post", prefix))?,
        })
    }
}

#[derive(Debug)]
pub struct WhisperTextDecoder {
    token_embedding: Tensor<f16>,
    positional_embedding: Tensor<f16>,
    ln: WhisperLayerNorm,
    layers: Vec<WhisperTransformerBlock>,
}

impl WhisperTextDecoder {
    pub fn new(pickle: &PickledModel<WhisperDims>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            token_embedding: load_tensor(pickle, &format!("{}.token_embedding.weight", prefix))?,
            positional_embedding: load_tensor(pickle, &format!("{}.positional_embedding", prefix))?,
            ln: WhisperLayerNorm::new(pickle, &format!("{}.ln", prefix))?,
            layers: (0..pickle.metadata.n_text_layer)
                .map(|i| {
                    WhisperTransformerBlock::new(pickle, &format!("{}.blocks.{}", prefix, i), true)
                })
                .collect::<anyhow::Result<Vec<_>>>()?,
        })
    }
}

pub struct WhisperModel {
    dims: WhisperDims,
    encoder: WhisperAudioEncoder,
    decoder: WhisperTextDecoder,
    tokenizer: CoreBPE,
}

impl Debug for WhisperModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("WhisperModel")
            .field("dims", &self.dims)
            .field("encoder", &self.encoder)
            .field("decoder", &self.decoder)
            .finish()
    }
}

impl WhisperModel {
    pub fn new(pickle: &PickledModel<WhisperDims>) -> anyhow::Result<Self> {
        Ok(Self {
            dims: pickle.metadata.clone(),
            encoder: WhisperAudioEncoder::new(pickle, "encoder")?,
            decoder: WhisperTextDecoder::new(pickle, "decoder")?,
            tokenizer: if pickle.metadata.n_vocab == 51865 {
                tiktoken_rs::whisper_multilingual()?
            } else {
                tiktoken_rs::whisper_gpt2()?
            },
        })
    }

    pub fn tokenizer(&self) -> &CoreBPE {
        &self.tokenizer
    }

    pub fn dims(&self) -> &WhisperDims {
        &self.dims
    }
}

fn sinusoid_position_embedding(dims: &WhisperDims) -> Tensor<f16> {
    let mut pos_embedding_vec =
        vec![f16::from_f32(0.0); dims.n_audio_ctx as usize * dims.n_audio_state as usize];
    let increment = (10000.0f32).ln() / (dims.n_audio_state / 2 - 1) as f32;
    for i in 0..dims.n_audio_ctx as usize {
        for j in 0..dims.n_audio_state as usize / 2 {
            let theta = i as f32 * (j as f32 * -increment).exp();
            pos_embedding_vec[i * dims.n_audio_state as usize + j] = f16::from_f32(theta.sin());
            pos_embedding_vec
                [i * dims.n_audio_state as usize + dims.n_audio_state as usize / 2 + j] =
                f16::from_f32(theta.cos());
        }
    }
    let pos_embedding = Tensor::from_vec(
        pos_embedding_vec,
        TensorLayout::row_major(&[dims.n_audio_ctx as usize, dims.n_audio_state as usize]),
    );
    pos_embedding
}

struct WhisperContextCacheLayer {
    key: Tensor<f16>,
    value: Tensor<f16>,
}

pub struct WhisperContext {
    model: Arc<WhisperModel>,
    kernels: Arc<dyn CommonKernels>,
    mel_transform: mel::LogMelSpectrogramTransform,
    kv_cache: Vec<WhisperContextCacheLayer>,
}

impl WhisperContext {
    pub fn new(model: Arc<WhisperModel>, kernels: Arc<dyn CommonKernels>) -> anyhow::Result<Self> {
        Ok(Self {
            kernels,
            mel_transform: mel::LogMelSpectrogramTransform::new(
                WHISPER_N_FFT,
                WHISPER_N_MELS,
                WHISPER_HOP_LENGTH,
                WHISPER_SAMPLE_RATE as f32,
            ),
            kv_cache: (0..model.dims.n_text_layer)
                .map(|_| {
                    Ok(WhisperContextCacheLayer {
                        key: Tensor::new_hip(&[
                            model.dims.n_text_ctx as usize,
                            model.dims.n_text_state as usize,
                        ])?,
                        value: Tensor::new_hip(&[
                            model.dims.n_text_ctx as usize,
                            model.dims.n_text_state as usize,
                        ])?,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?,
            model,
        })
    }

    pub fn model(&self) -> &WhisperModel {
        &self.model
    }

    fn process_layer(
        &self, layer: &WhisperTransformerBlock, hidden_state: &mut TensorViewMut<f16>,
        features: Option<&TensorView<f16>>, mask: MatmulMask,
    ) -> anyhow::Result<()> {
        let mut ln_out = Tensor::new_hip(&hidden_state.layout().dims)?;
        let mut mlp_hidden =
            Tensor::new_hip(&[ln_out.size(-2) as usize, ln_out.size(-1) as usize * 4])?;
        self.kernels.layer_norm(
            &mut ln_out.as_view_mut(),
            &hidden_state.as_view(),
            &layer.attn_ln.weight.as_view(),
            &layer.attn_ln.bias.as_view(),
            1.0e-5,
        )?;
        self.residual_attention(
            hidden_state,
            &ln_out.as_view(),
            &ln_out.as_view(),
            &layer.attn,
            mask,
        )?;
        if let Some(cross_attn) = layer.cross_attn.as_ref() {
            let cross_attn_ln = layer.cross_attn_ln.as_ref().unwrap();
            self.kernels.layer_norm(
                &mut ln_out.as_view_mut(),
                &hidden_state.as_view(),
                &cross_attn_ln.weight.as_view(),
                &cross_attn_ln.bias.as_view(),
                1.0e-5,
            )?;
            self.residual_attention(
                hidden_state,
                &ln_out.as_view(),
                features.expect("encoded features expected for cross attention layer"),
                cross_attn,
                MatmulMask::None,
            )?;
        }
        self.kernels.layer_norm(
            &mut ln_out.as_view_mut(),
            &hidden_state.as_view(),
            &layer.mlp_ln.weight.as_view(),
            &layer.mlp_ln.bias.as_view(),
            1.0e-5,
        )?;
        self.kernels.matmul_f16_fast(
            &mut mlp_hidden.as_view_mut(),
            &ln_out.as_view(),
            &layer.mlp_0.weight.as_view().permute(&[0, 1, 3, 2]),
            MatmulOptions::new().store(MatmulStore::BetaGeluBias(0.0, &layer.mlp_0.bias.as_view())),
        )?;
        self.kernels.matmul_f16_fast(
            hidden_state,
            &mlp_hidden.as_view(),
            &layer.mlp_2.weight.as_view().permute(&[0, 1, 3, 2]),
            MatmulOptions::new().store(MatmulStore::BetaBias(1.0, &layer.mlp_2.bias.as_view())),
        )?;
        Ok(())
    }

    fn residual_attention(
        &self, hidden_state: &mut TensorViewMut<f16>, ln_out: &TensorView<f16>,
        kv_input: &TensorView<f16>, attn: &WhisperAttention, mask: MatmulMask,
    ) -> Result<(), anyhow::Error> {
        // TODO this incidentally works but should reference the right hparam
        let heads = self.model.dims.n_audio_head as isize;
        let mut query = Tensor::new_hip(&ln_out.layout().dims)?;
        let mut key = Tensor::new_hip(&kv_input.layout().dims)?;
        let mut value = Tensor::new_hip(&kv_input.layout().dims)?;
        let mut qkv = Tensor::new_hip(&ln_out.layout().dims)?;
        self.kernels.matmul_f16_fast(
            &mut query.as_view_mut(),
            ln_out,
            &attn.query.weight.as_view().permute(&[0, 1, 3, 2]),
            MatmulOptions::new().store(MatmulStore::BetaBias(0.0, &attn.query.bias.as_view())),
        )?;
        self.kernels.matmul_f16_fast(
            &mut key.as_view_mut(),
            kv_input,
            &attn.key.as_view().permute(&[0, 1, 3, 2]),
            MatmulOptions::new(),
        )?;
        self.kernels.matmul_f16_fast(
            &mut value.as_view_mut(),
            kv_input,
            &attn.value.weight.as_view().permute(&[0, 1, 3, 2]),
            MatmulOptions::new().store(MatmulStore::BetaBias(0.0, &attn.value.bias.as_view())),
        )?;
        let q_view =
            query.as_view().shape_cast(&[query.size(-2) as isize, heads, -1]).permute(&[1, 0, 2]);
        let k_view =
            key.as_view().shape_cast(&[key.size(-2) as isize, heads, -1]).permute(&[1, 2, 0]);
        let v_view =
            value.as_view().shape_cast(&[value.size(-2) as isize, heads, -1]).permute(&[1, 0, 2]);
        let mut qk = Tensor::new_hip(&[heads as usize, q_view.size(-2), k_view.size(-1)])?;
        self.kernels.matmul_f16_fast(
            &mut qk.as_view_mut(),
            &q_view,
            &k_view,
            MatmulOptions::new()
                .load(MatmulLoad::Scale(
                    // TODO this incidentally works but should reference the right hparam
                    (self.model.dims.n_audio_state as f32 / self.model.dims.n_audio_head as f32)
                        .powf(-0.25),
                ))
                .mask(mask),
        )?;
        self.kernels.softmax_rows_inplace(&mut qk.as_view_mut(), 1.0)?;
        let mut qkv_view =
            qkv.as_view_mut().shape_cast(&[query.size(-2) as isize, heads, -1]).permute(&[1, 0, 2]);
        self.kernels.matmul_f16_fast(
            &mut qkv_view,
            &qk.as_view(),
            &v_view,
            MatmulOptions::new(),
        )?;
        self.kernels.matmul_f16_fast(
            hidden_state,
            &qkv.as_view(),
            &attn.out.weight.as_view().permute(&[0, 1, 3, 2]),
            MatmulOptions::new().store(MatmulStore::BetaBias(1.0, &attn.out.bias.as_view())),
        )?;
        Ok(())
    }

    pub fn encode(&mut self, wave: &[f32]) -> anyhow::Result<Tensor<f16>> {
        // TODO we should be accepting a fixed length and not allocate these each time
        let mut complex_scratch =
            vec![Complex32::new(0.0, 0.0); self.mel_transform.complex_scratch_size(wave.len())];
        let mut real_scratch = vec![0.0; self.mel_transform.real_scratch_size(wave.len())];
        let mut mel_spec = vec![0.0; self.mel_transform.output_size(wave.len())];
        self.mel_transform.process(&mut mel_spec, wave, &mut complex_scratch, &mut real_scratch);
        let mut mels_half = vec![f16::from_f32(0.0); mel_spec.len()];
        for (l, r) in mels_half.iter_mut().zip(mel_spec.iter()) {
            *l = f16::from_f32(*r);
        }
        let mels_half = Tensor::from_vec(
            mels_half,
            TensorLayout::row_major(&[WHISPER_N_MELS, self.mel_transform.num_cols(wave.len())]),
        )
        .into_hip()?;
        let mut conv_out =
            Tensor::new_hip(&[self.model.dims.n_audio_state as usize, mels_half.size(-1)])?;
        let mut hidden_state =
            Tensor::new_hip(&[self.model.dims.n_audio_state as usize, mels_half.size(-1) / 2])?;
        self.kernels.conv1d(
            &mut conv_out.as_view_mut(),
            &mels_half.as_view(),
            &self.model.encoder.conv1.weight.as_view(),
            &self.model.encoder.conv1.bias.as_view(),
            3,
            1,
            1,
            Conv1dActivation::Gelu,
        )?;
        self.kernels.conv1d(
            &mut hidden_state.as_view_mut(),
            &conv_out.as_view(),
            &self.model.encoder.conv2.weight.as_view(),
            &self.model.encoder.conv2.bias.as_view(),
            3,
            2,
            1,
            Conv1dActivation::Gelu,
        )?;
        let mut hidden_state = hidden_state.as_view_mut().permute(&[1, 0]);
        // TODO this can be fused
        self.kernels.elementwise_binary_2d_f16_inplace(
            &mut hidden_state,
            &self.model.encoder.position_embedding.as_view(),
            BinaryOp::Add,
        )?;
        for layer in &self.model.encoder.layers {
            self.process_layer(layer, &mut hidden_state, None, MatmulMask::None)?;
        }
        let mut features = Tensor::new_hip(&[
            self.model.dims.n_audio_ctx as usize,
            self.model.dims.n_audio_state as usize,
        ])?;
        self.kernels.layer_norm(
            &mut features.as_view_mut(),
            &hidden_state.as_view(),
            &self.model.encoder.ln_post.weight.as_view(),
            &self.model.encoder.ln_post.bias.as_view(),
            1.0e-5,
        )?;
        Ok(features)
    }

    pub fn decode(
        &mut self, features: TensorView<f16>, tokens: &[i32],
    ) -> anyhow::Result<Tensor<f16>> {
        let mut hidden_state =
            Tensor::new_hip(&[tokens.len(), self.model.dims.n_text_state as usize])?;
        let mut ln_out = Tensor::new_hip(&[tokens.len(), self.model.dims.n_text_state as usize])?;
        let tokens_gpu =
            Tensor::from_vec(tokens.into(), TensorLayout::row_major(&[tokens.len()])).into_hip()?;
        let mut logits = Tensor::new_hip(&[tokens.len(), self.model.dims.n_vocab as usize])?;
        self.kernels.embed(
            &mut hidden_state.as_view_mut(),
            tokens_gpu.as_view(),
            self.model.decoder.token_embedding.as_view(),
        )?;
        self.kernels.elementwise_binary_2d_f16_inplace(
            &mut hidden_state.as_view_mut(),
            &self.model.decoder.positional_embedding.as_view(),
            BinaryOp::Add,
        )?;
        for layer in &self.model.decoder.layers {
            self.process_layer(
                layer,
                &mut hidden_state.as_view_mut(),
                Some(&features),
                MatmulMask::Causal,
            )?;
        }
        self.kernels.layer_norm(
            &mut ln_out.as_view_mut(),
            &hidden_state.as_view(),
            &self.model.decoder.ln.weight.as_view(),
            &self.model.decoder.ln.bias.as_view(),
            1.0e-5,
        )?;
        self.kernels.matmul_f16_fast(
            &mut logits.as_view_mut(),
            &ln_out.as_view(),
            &self.model.decoder.token_embedding.as_view().permute(&[0, 1, 3, 2]),
            MatmulOptions::new(),
        )?;
        Ok(logits)
    }
}
