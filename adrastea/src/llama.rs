use alloc::sync::Arc;
use half::f16;
use sentencepiece::SentencePieceProcessor;
use serde::Deserialize;

use crate::{
    kernels::{BinaryOp, CommonKernels, MatmulMask, MatmulOptions, MatmulStore},
    pickle::{load_tensor, PickledModel},
    tensor::{Tensor, TensorLayout, TensorViewMut},
    util::round_up,
};

// TODO: last logits, kv cache, bring over many optimizations =(
// bring back multi-gpu and quantization =(

pub struct LlamaAttention {
    query: Tensor<f16>,
    key: Tensor<f16>,
    value: Tensor<f16>,
    out: Tensor<f16>,
}

impl LlamaAttention {
    pub fn new(pickle: &PickledModel<()>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            query: load_tensor(pickle, &format!("{}.wq.weight", prefix))?,
            key: load_tensor(pickle, &format!("{}.wk.weight", prefix))?,
            value: load_tensor(pickle, &format!("{}.wv.weight", prefix))?,
            out: load_tensor(pickle, &format!("{}.wo.weight", prefix))?,
        })
    }
}

pub struct LlamaFeedForward {
    w1: Tensor<f16>,
    w2: Tensor<f16>,
    w3: Tensor<f16>,
}

impl LlamaFeedForward {
    pub fn new(pickle: &PickledModel<()>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            w1: load_tensor(pickle, &format!("{}.w1.weight", prefix))?,
            w2: load_tensor(pickle, &format!("{}.w2.weight", prefix))?,
            w3: load_tensor(pickle, &format!("{}.w3.weight", prefix))?,
        })
    }
}

pub struct LlamaTransformerBlock {
    attn_norm: Tensor<f16>,
    attn: LlamaAttention,
    ffn_norm: Tensor<f16>,
    ffn: LlamaFeedForward,
}

impl LlamaTransformerBlock {
    pub fn new(pickle: &PickledModel<()>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            attn_norm: load_tensor(pickle, &format!("{}.attention_norm.weight", prefix))?,
            attn: LlamaAttention::new(pickle, &format!("{}.attention", prefix))?,
            ffn_norm: load_tensor(pickle, &format!("{}.ffn_norm.weight", prefix))?,
            ffn: LlamaFeedForward::new(pickle, &format!("{}.feed_forward", prefix))?,
        })
    }
}

#[derive(Deserialize)]
pub struct LlamaParams {
    pub dim: u32,
    pub multiple_of: u32,
    pub n_heads: u32,
    pub n_layers: u32,
    pub norm_eps: f32,
    pub vocab_size: isize,
}

impl LlamaParams {
    pub fn ffn_dim(&self) -> usize {
        round_up(self.dim * 8 / 3, self.multiple_of) as usize
    }
}

pub struct LlamaModel {
    params: LlamaParams,
    layers: Vec<LlamaTransformerBlock>,
    output: Tensor<f16>,
    norm: Tensor<f16>,
    tok_embeddings: Tensor<f16>,
    tokenizer: SentencePieceProcessor,
}

impl LlamaModel {
    pub fn new(
        pickle: &PickledModel<()>, mut params: LlamaParams, tokenizer: SentencePieceProcessor,
    ) -> anyhow::Result<Self> {
        params.vocab_size = tokenizer.len() as isize;
        Ok(Self {
            layers: (0..params.n_layers)
                .map(|i| LlamaTransformerBlock::new(pickle, &format!("layers.{}", i)))
                .collect::<anyhow::Result<_>>()?,
            params,
            tokenizer,
            output: load_tensor(pickle, "output.weight")?,
            norm: load_tensor(pickle, "norm.weight")?,
            tok_embeddings: load_tensor(pickle, "tok_embeddings.weight")?,
        })
    }

    pub fn params(&self) -> &LlamaParams {
        &self.params
    }

    pub fn tokenizer(&self) -> &SentencePieceProcessor {
        &self.tokenizer
    }
}

pub struct LlamaContext {
    model: Arc<LlamaModel>,
    kernels: Arc<dyn CommonKernels>,
}

impl LlamaContext {
    pub fn new(model: Arc<LlamaModel>, kernels: Arc<dyn CommonKernels>) -> Self {
        Self { model, kernels }
    }

    pub fn model(&self) -> &Arc<LlamaModel> {
        &self.model
    }

    pub fn kernels(&self) -> &Arc<dyn CommonKernels> {
        &self.kernels
    }

    pub fn decode(&mut self, tokens: &[i32]) -> anyhow::Result<Tensor<f16>> {
        let mut hidden_state = Tensor::new_hip(&[tokens.len(), self.model.params.dim as usize])?;
        let mut normed_state = Tensor::new_hip(&hidden_state.layout().dims)?;
        let tokens_gpu =
            Tensor::from_vec(tokens.into(), TensorLayout::row_major(&[tokens.len()])).into_hip()?;
        let mut logits = Tensor::new_hip(&[tokens.len(), self.model.params.vocab_size as usize])?;
        self.kernels.embed(
            &mut hidden_state.as_view_mut(),
            tokens_gpu.as_view(),
            self.model.tok_embeddings.as_view(),
        )?;

        for layer in &self.model.layers {
            self.process_layer(&mut hidden_state.as_view_mut(), layer)?;
        }
        self.kernels.rms_norm(
            &mut normed_state.as_view_mut(),
            &hidden_state.as_view(),
            &self.model.norm.as_view(),
            self.model.params.norm_eps,
        )?;
        self.kernels.matmul_f16(
            &mut logits.as_view_mut(),
            &normed_state.as_view(),
            &self.model.output.as_view().permute(&[1, 0]),
            MatmulOptions::new(),
        )?;
        Ok(logits)
    }

    fn process_layer(
        &self, hidden_state: &mut TensorViewMut<f16>, layer: &LlamaTransformerBlock,
    ) -> anyhow::Result<()> {
        let mut normed_state = Tensor::new_hip(&hidden_state.layout().dims)?;
        let mut query = Tensor::new_hip(&hidden_state.layout().dims)?;
        let mut key = Tensor::new_hip(&hidden_state.layout().dims)?;
        let mut value = Tensor::new_hip(&hidden_state.layout().dims)?;
        let mut qkv = Tensor::new_hip(&hidden_state.layout().dims)?;
        let mut ffn_w1 =
            Tensor::new_hip(&[hidden_state.size(-2), self.model.params.ffn_dim() as usize])?;
        let mut ffn_w3 =
            Tensor::new_hip(&[hidden_state.size(-2), self.model.params.ffn_dim() as usize])?;
        self.kernels.rms_norm(
            &mut normed_state.as_view_mut(),
            &hidden_state.as_view(),
            &layer.attn_norm.as_view(),
            self.model.params.norm_eps,
        )?;
        self.kernels.matmul_f16(
            &mut query.as_view_mut(),
            &normed_state.as_view(),
            &layer.attn.query.as_view().permute(&[1, 0]),
            MatmulOptions::new(),
        )?;
        self.kernels.rotary_inplace(
            &mut query.as_view_mut(),
            self.model.params.n_heads as i32,
            0,
            10000.0,
        )?;
        self.kernels.matmul_f16(
            &mut key.as_view_mut(),
            &normed_state.as_view(),
            &layer.attn.key.as_view().permute(&[1, 0]),
            MatmulOptions::new(),
        )?;
        self.kernels.rotary_inplace(
            &mut key.as_view_mut(),
            self.model.params.n_heads as i32,
            0,
            10000.0,
        )?;
        self.kernels.matmul_f16(
            &mut value.as_view_mut(),
            &normed_state.as_view(),
            &layer.attn.value.as_view().permute(&[1, 0]),
            MatmulOptions::new(),
        )?;
        let heads = self.model.params.n_heads as isize;
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
                    1.0 / (self.model.params.dim as f32 / self.model.params.n_heads as f32).sqrt(),
                ))
                .mask(MatmulMask::Causal),
        )?;
        self.kernels.softmax_rows_inplace(&mut qk.as_view_mut(), 1.0)?;
        let mut qkv_view =
            qkv.as_view_mut().shape_cast(&[query.size(-2) as isize, heads, -1]).permute(&[1, 0, 2]);
        self.kernels.matmul_f16(&mut qkv_view, &qk.as_view(), &v_view, MatmulOptions::new())?;
        self.kernels.matmul_f16(
            hidden_state,
            &qkv.as_view(),
            &layer.attn.out.as_view().permute(&[1, 0]),
            MatmulOptions::new().store(MatmulStore::Add),
        )?;
        self.kernels.rms_norm(
            &mut normed_state.as_view_mut(),
            &hidden_state.as_view(),
            &layer.ffn_norm.as_view(),
            self.model.params.norm_eps,
        )?;
        self.kernels.matmul_f16(
            &mut ffn_w1.as_view_mut(),
            &normed_state.as_view(),
            &layer.ffn.w1.as_view().permute(&[1, 0]),
            MatmulOptions::new(),
        )?;
        self.kernels.matmul_f16(
            &mut ffn_w3.as_view_mut(),
            &normed_state.as_view(),
            &layer.ffn.w3.as_view().permute(&[1, 0]),
            MatmulOptions::new(),
        )?;
        self.kernels.elementwise_binary_2d_f16_inplace(
            &mut ffn_w1.as_view_mut(),
            &ffn_w3.as_view(),
            BinaryOp::SiluMul,
        )?;
        self.kernels.matmul_f16(
            hidden_state,
            &ffn_w1.as_view(),
            &layer.ffn.w2.as_view().permute(&[1, 0]),
            MatmulOptions::new().store(MatmulStore::Add),
        )?;
        Ok(())
    }
}
