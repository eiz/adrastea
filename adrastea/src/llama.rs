use alloc::sync::Arc;
use half::f16;
use serde::Deserialize;

use crate::{
    kernels::CommonKernels,
    pickle::{load_tensor, PickledModel},
    tensor::Tensor,
};

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

pub struct LlamaModel {
    layers: Vec<LlamaTransformerBlock>,
    output: Tensor<f16>,
    norm: Tensor<f16>,
    tok_embeddings: Tensor<f16>,
}

impl LlamaModel {
    pub fn new(pickle: &PickledModel<()>, params: LlamaParams) -> anyhow::Result<Self> {
        Ok(Self {
            layers: (0..params.n_layers)
                .map(|i| LlamaTransformerBlock::new(pickle, &format!("layers.{}", i)))
                .collect::<anyhow::Result<_>>()?,
            output: load_tensor(pickle, "output.weight")?,
            norm: load_tensor(pickle, "norm.weight")?,
            tok_embeddings: load_tensor(pickle, "tok_embeddings.weight")?,
        })
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

    pub fn decode(&mut self, tokens: &[i32]) -> anyhow::Result<Tensor<f16>> {
        todo!()
    }
}
