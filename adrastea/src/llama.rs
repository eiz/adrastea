use alloc::sync::Arc;
use half::f16;
use sentencepiece::SentencePieceProcessor;
use serde::Deserialize;

use crate::{
    kernels::{BinaryOp, CommonKernels, MatmulMask, MatmulOptions, MatmulStore, UnaryOp},
    pickle::{load_tensor, PickledModel, ShardedModel},
    tensor::{Tensor, TensorLayout, TensorView, TensorViewMut},
    util::round_up,
};

// TODO: last logits, kv cache, bring over many optimizations =(
// bring back multi-gpu and quantization =(

#[derive(Copy, Clone, Debug)]
pub enum LlamaAttentionAddress {
    Query,
    Key,
    Value,
    Out,
}

#[derive(Copy, Clone, Debug)]
pub enum LlamaMLPAddress {
    Expand,
    Contract,
    Gate,
}

#[derive(Copy, Clone, Debug)]
pub enum LlamaTransformerBlockAddress {
    AttentionNorm,
    Attention(LlamaAttentionAddress),
    MLPNorm,
    MLP(LlamaMLPAddress),
}

#[derive(Copy, Clone, Debug)]
pub enum LlamaModelAddress {
    TokenEmbedding,
    OutputNorm,
    TokenProjection,
    Layer(usize, LlamaTransformerBlockAddress),
}

pub trait ResolveTensorAddress<T> {
    fn resolve_tensor_address(address: T) -> String;
}

pub trait LoadTensor<T> {
    fn load_tensor(&self, address: T) -> anyhow::Result<Tensor<f16>>;
}

pub struct HuggingFaceTensorNames;

impl ResolveTensorAddress<LlamaModelAddress> for HuggingFaceTensorNames {
    fn resolve_tensor_address(address: LlamaModelAddress) -> String {
        match address {
            LlamaModelAddress::TokenEmbedding => "model.embed_tokens.weight".into(),
            LlamaModelAddress::OutputNorm => "model.norm.weight".into(),
            LlamaModelAddress::TokenProjection => "lm_head.weight".into(),
            LlamaModelAddress::Layer(n, address) => {
                format!("model.layers.{n}.{}", Self::resolve_tensor_address(address))
            }
        }
    }
}

impl ResolveTensorAddress<LlamaTransformerBlockAddress> for HuggingFaceTensorNames {
    fn resolve_tensor_address(address: LlamaTransformerBlockAddress) -> String {
        match address {
            LlamaTransformerBlockAddress::Attention(address) => {
                format!("self_attn.{}", Self::resolve_tensor_address(address))
            }
            LlamaTransformerBlockAddress::MLP(address) => {
                format!("mlp.{}", Self::resolve_tensor_address(address))
            }
            LlamaTransformerBlockAddress::AttentionNorm => "input_layernorm.weight".into(),
            LlamaTransformerBlockAddress::MLPNorm => "post_attention_layernorm.weight".into(),
        }
    }
}

impl ResolveTensorAddress<LlamaAttentionAddress> for HuggingFaceTensorNames {
    fn resolve_tensor_address(address: LlamaAttentionAddress) -> String {
        match address {
            LlamaAttentionAddress::Query => "q_proj.weight".into(),
            LlamaAttentionAddress::Key => "k_proj.weight".into(),
            LlamaAttentionAddress::Value => "v_proj.weight".into(),
            LlamaAttentionAddress::Out => "o_proj.weight".into(),
        }
    }
}

impl ResolveTensorAddress<LlamaMLPAddress> for HuggingFaceTensorNames {
    fn resolve_tensor_address(address: LlamaMLPAddress) -> String {
        match address {
            LlamaMLPAddress::Expand => "up_proj.weight".into(),
            LlamaMLPAddress::Contract => "down_proj.weight".into(),
            LlamaMLPAddress::Gate => "gate_proj.weight".into(),
        }
    }
}

pub struct MetaTensorNames;

impl ResolveTensorAddress<LlamaModelAddress> for MetaTensorNames {
    fn resolve_tensor_address(address: LlamaModelAddress) -> String {
        match address {
            LlamaModelAddress::TokenEmbedding => "tok_embeddings.weight".into(),
            LlamaModelAddress::OutputNorm => "norm.weight".into(),
            LlamaModelAddress::TokenProjection => "output.weight".into(),
            LlamaModelAddress::Layer(n, address) => {
                format!("layers.{n}.{}", Self::resolve_tensor_address(address))
            }
        }
    }
}

impl ResolveTensorAddress<LlamaTransformerBlockAddress> for MetaTensorNames {
    fn resolve_tensor_address(address: LlamaTransformerBlockAddress) -> String {
        match address {
            LlamaTransformerBlockAddress::Attention(address) => {
                format!("attention.{}", Self::resolve_tensor_address(address))
            }
            LlamaTransformerBlockAddress::MLP(address) => {
                format!("feed_forward.{}", Self::resolve_tensor_address(address))
            }
            LlamaTransformerBlockAddress::AttentionNorm => "attention_norm.weight".into(),
            LlamaTransformerBlockAddress::MLPNorm => "ffn_norm.weight".into(),
        }
    }
}

impl ResolveTensorAddress<LlamaAttentionAddress> for MetaTensorNames {
    fn resolve_tensor_address(address: LlamaAttentionAddress) -> String {
        match address {
            LlamaAttentionAddress::Query => "wq.weight".into(),
            LlamaAttentionAddress::Key => "wk.weight".into(),
            LlamaAttentionAddress::Value => "wv.weight".into(),
            LlamaAttentionAddress::Out => "wo.weight".into(),
        }
    }
}

impl ResolveTensorAddress<LlamaMLPAddress> for MetaTensorNames {
    fn resolve_tensor_address(address: LlamaMLPAddress) -> String {
        match address {
            LlamaMLPAddress::Expand => "w3.weight".into(),
            LlamaMLPAddress::Contract => "w2.weight".into(),
            LlamaMLPAddress::Gate => "w1.weight".into(),
        }
    }
}

pub struct MetaLlamaModelLoader {
    model: PickledModel<()>,
}

impl MetaLlamaModelLoader {
    pub fn new(model: PickledModel<()>) -> Self {
        Self { model }
    }
}

impl LoadTensor<LlamaModelAddress> for MetaLlamaModelLoader {
    fn load_tensor(&self, address: LlamaModelAddress) -> anyhow::Result<Tensor<f16>> {
        let name = MetaTensorNames::resolve_tensor_address(address);
        let result = load_tensor(&self.model, &name)?;
        Ok(result)
    }
}

pub struct HuggingFaceLlamaModelLoader<'a> {
    model: &'a ShardedModel,
    params: &'a LlamaParams,
    kernels: &'a dyn CommonKernels,
}

impl<'a> HuggingFaceLlamaModelLoader<'a> {
    pub fn new(
        model: &'a ShardedModel, params: &'a LlamaParams, kernels: &'a dyn CommonKernels,
    ) -> Self {
        Self { model, params, kernels }
    }
}

impl<'a> LoadTensor<LlamaModelAddress> for HuggingFaceLlamaModelLoader<'a> {
    fn load_tensor(&self, address: LlamaModelAddress) -> anyhow::Result<Tensor<f16>> {
        let name = HuggingFaceTensorNames::resolve_tensor_address(address);
        let result = self.model.load_tensor(&name)?;
        match address {
            LlamaModelAddress::Layer(
                _,
                LlamaTransformerBlockAddress::Attention(
                    LlamaAttentionAddress::Key | LlamaAttentionAddress::Query,
                ),
            ) => {
                // I don't know what the attempt at cleverness was, but we have
                // to undo the permutation done to the QK matrices by the
                // Hugging Face conversion code
                // TODO: this code would be a lot simpler with an actual reshape operator!
                let mut unswizzled_result =
                    Tensor::new_hip(&[self.params.dim as usize, self.params.dim as usize])?;
                self.kernels.elementwise_unary_2d_f16(
                    &mut unswizzled_result.as_view_mut().shape_cast(&[
                        self.params.n_heads as isize,
                        (self.params.dim / self.params.n_heads / 2) as isize,
                        2,
                        self.params.dim as isize,
                    ]),
                    &result
                        .as_view()
                        .shape_cast(&[
                            self.params.n_heads as isize,
                            2,
                            (self.params.dim / self.params.n_heads / 2) as isize,
                            self.params.dim as isize,
                        ])
                        .permute(&[0, 2, 1, 3]),
                    UnaryOp::Identity,
                )?;
                Ok(unswizzled_result)
            }
            _ => Ok(result),
        }
    }
}

struct LlamaTransformerBlockLoader<'a, T: LoadTensor<LlamaModelAddress>> {
    inner: &'a T,
    layer: usize,
}

impl<'a, T: LoadTensor<LlamaModelAddress>> LlamaTransformerBlockLoader<'a, T> {
    fn new(inner: &'a T, layer: usize) -> Self {
        Self { inner, layer }
    }
}

impl<'a, T: LoadTensor<LlamaModelAddress>> LoadTensor<LlamaTransformerBlockAddress>
    for LlamaTransformerBlockLoader<'a, T>
{
    fn load_tensor(&self, address: LlamaTransformerBlockAddress) -> anyhow::Result<Tensor<f16>> {
        self.inner.load_tensor(LlamaModelAddress::Layer(self.layer, address))
    }
}

impl<'a, T: LoadTensor<LlamaModelAddress>> LoadTensor<LlamaAttentionAddress>
    for LlamaTransformerBlockLoader<'a, T>
{
    fn load_tensor(&self, address: LlamaAttentionAddress) -> anyhow::Result<Tensor<f16>> {
        self.inner.load_tensor(LlamaModelAddress::Layer(
            self.layer,
            LlamaTransformerBlockAddress::Attention(address),
        ))
    }
}

impl<'a, T: LoadTensor<LlamaModelAddress>> LoadTensor<LlamaMLPAddress>
    for LlamaTransformerBlockLoader<'a, T>
{
    fn load_tensor(&self, address: LlamaMLPAddress) -> anyhow::Result<Tensor<f16>> {
        self.inner.load_tensor(LlamaModelAddress::Layer(
            self.layer,
            LlamaTransformerBlockAddress::MLP(address),
        ))
    }
}

pub struct LlamaAttention {
    query: Tensor<f16>,
    key: Tensor<f16>,
    value: Tensor<f16>,
    out: Tensor<f16>,
}

impl LlamaAttention {
    pub fn new<T: LoadTensor<LlamaAttentionAddress>>(loader: &T) -> anyhow::Result<Self> {
        Ok(Self {
            query: loader.load_tensor(LlamaAttentionAddress::Query)?,
            key: loader.load_tensor(LlamaAttentionAddress::Key)?,
            value: loader.load_tensor(LlamaAttentionAddress::Value)?,
            out: loader.load_tensor(LlamaAttentionAddress::Out)?,
        })
    }
}

pub struct LlamaFeedForward {
    gate: Tensor<f16>,
    contract: Tensor<f16>,
    expand: Tensor<f16>,
}

impl LlamaFeedForward {
    pub fn new<T: LoadTensor<LlamaMLPAddress>>(loader: &T) -> anyhow::Result<Self> {
        Ok(Self {
            gate: loader.load_tensor(LlamaMLPAddress::Gate)?,
            contract: loader.load_tensor(LlamaMLPAddress::Contract)?,
            expand: loader.load_tensor(LlamaMLPAddress::Expand)?,
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
    pub fn new<
        T: LoadTensor<LlamaAttentionAddress>
            + LoadTensor<LlamaMLPAddress>
            + LoadTensor<LlamaTransformerBlockAddress>,
    >(
        loader: &T,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            attn_norm: loader.load_tensor(LlamaTransformerBlockAddress::AttentionNorm)?,
            ffn_norm: loader.load_tensor(LlamaTransformerBlockAddress::MLPNorm)?,
            attn: LlamaAttention::new(loader)?,
            ffn: LlamaFeedForward::new(loader)?,
        })
    }
}

#[derive(Clone, Deserialize)]
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
    pub fn new<T: LoadTensor<LlamaModelAddress>>(
        loader: &T, mut params: LlamaParams, tokenizer: SentencePieceProcessor, added_tokens: usize,
    ) -> anyhow::Result<Self> {
        params.vocab_size = (tokenizer.len() + added_tokens) as isize;
        Ok(Self {
            layers: (0..params.n_layers)
                .map(|i| {
                    LlamaTransformerBlock::new(&LlamaTransformerBlockLoader::new(
                        loader, i as usize,
                    ))
                })
                .collect::<anyhow::Result<_>>()?,
            params,
            tokenizer,
            output: loader.load_tensor(LlamaModelAddress::TokenProjection)?,
            norm: loader.load_tensor(LlamaModelAddress::OutputNorm)?,
            tok_embeddings: loader.load_tensor(LlamaModelAddress::TokenEmbedding)?,
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

    pub fn embed(
        &mut self, embedded: &mut TensorViewMut<f16>, tokens: &[i32],
    ) -> anyhow::Result<()> {
        let tokens_gpu =
            Tensor::from_vec(tokens.into(), TensorLayout::row_major(&[tokens.len()])).into_hip()?;
        self.kernels.embed(embedded, tokens_gpu.as_view(), self.model.tok_embeddings.as_view())
    }

    // TODO: this function should not exist in its current form but we are
    // deferring implementing kv cache for a little while longer
    pub fn decode_embedded(&mut self, embedded: &TensorView<f16>) -> anyhow::Result<Tensor<f16>> {
        let mut hidden_state = Tensor::new_hip(&embedded.layout().dims)?;
        let mut normed_state = Tensor::new_hip(&hidden_state.layout().dims)?;
        let mut logits =
            Tensor::new_hip(&[embedded.size(-2) as usize, self.model.params.vocab_size as usize])?;
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

    pub fn decode(&mut self, tokens: &[i32]) -> anyhow::Result<Tensor<f16>> {
        let mut hidden_state = Tensor::new_hip(&[tokens.len(), self.model.params.dim as usize])?;
        let mut normed_state = Tensor::new_hip(&hidden_state.layout().dims)?;
        let tokens_gpu =
            Tensor::from_vec(tokens.into(), TensorLayout::row_major(&[tokens.len()])).into_hip()?;
        self.kernels.embed(
            &mut hidden_state.as_view_mut(),
            tokens_gpu.as_view(),
            self.model.tok_embeddings.as_view(),
        )?;
        let mut logits = Tensor::new_hip(&[tokens.len(), self.model.params.vocab_size as usize])?;
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
        let mut gate_in =
            Tensor::new_hip(&[hidden_state.size(-2), self.model.params.ffn_dim() as usize])?;
        let mut expand =
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
            &mut gate_in.as_view_mut(),
            &normed_state.as_view(),
            &layer.ffn.gate.as_view().permute(&[1, 0]),
            MatmulOptions::new(),
        )?;
        self.kernels.matmul_f16(
            &mut expand.as_view_mut(),
            &normed_state.as_view(),
            &layer.ffn.expand.as_view().permute(&[1, 0]),
            MatmulOptions::new(),
        )?;
        self.kernels.elementwise_binary_2d_f16_inplace(
            &mut gate_in.as_view_mut(),
            &expand.as_view(),
            BinaryOp::SiluMul,
        )?;
        self.kernels.matmul_f16(
            hidden_state,
            &gate_in.as_view(),
            &layer.ffn.contract.as_view().permute(&[1, 0]),
            MatmulOptions::new().store(MatmulStore::Add),
        )?;
        Ok(())
    }
}
