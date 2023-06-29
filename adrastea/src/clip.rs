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

use std::{fs::File, io::Write, path::Path};

use alloc::sync::Arc;
use half::f16;
use memmap2::Mmap;
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

fn to_f16(kernels: &dyn CommonKernels, tensor: Tensor<f32>) -> anyhow::Result<Tensor<f16>> {
    let mut output = Tensor::new_hip_layout(tensor.layout().clone())?;
    kernels.fp32_to_fp16(&mut output.as_view_mut(), &tensor.as_view())?;
    Ok(output)
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
    pub fn new(builder: &ClipModelBuilder, prefix: &str) -> anyhow::Result<Self> {
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
    pub fn new(builder: &ClipModelBuilder, prefix: &str) -> anyhow::Result<Self> {
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
    pub fn new(builder: &ClipModelBuilder, prefix: &str) -> anyhow::Result<Self> {
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
    pub fn new(builder: &ClipModelBuilder, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            fc1: ClipLinear::new(builder, &format!("{}.fc1", prefix))?,
            fc2: ClipLinear::new(builder, &format!("{}.fc2", prefix))?,
        })
    }
}

struct ClipVisionTransformerBlock {
    layer_norm1: ClipLayerNorm,
    attn: ClipAttention,
    layer_norm2: ClipLayerNorm,
    mlp: ClipMLP,
}

impl ClipVisionTransformerBlock {
    pub fn new(builder: &ClipModelBuilder, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            layer_norm1: ClipLayerNorm::new(builder, &format!("{}.layer_norm1", prefix))?,
            attn: ClipAttention::new(builder, &format!("{}.self_attn", prefix))?,
            layer_norm2: ClipLayerNorm::new(builder, &format!("{}.layer_norm2", prefix))?,
            mlp: ClipMLP::new(builder, &format!("{}.mlp", prefix))?,
        })
    }
}

struct ClipModelBuilder<'a> {
    pickle: &'a PickledModel<()>,
    kernels: &'a dyn CommonKernels,
    params: &'a ClipParams,
}

impl<'a> ClipModelBuilder<'a> {
    pub fn new(
        pickle: &'a PickledModel<()>, kernels: &'a dyn CommonKernels, params: &'a ClipParams,
    ) -> Self {
        Self { pickle, kernels, params }
    }

    pub fn load_tensor_f16(&self, name: &str) -> anyhow::Result<Tensor<f16>> {
        to_f16(self.kernels, load_tensor(self.pickle, name)?)
    }
}

struct ClipVisionTransformer {
    params: ClipVisionParams,
    class_embedding: Tensor<f16>,
    patch_embedding: Tensor<f16>,
    pre_layernorm: ClipLayerNorm,
    position_embedding: Tensor<f16>,
    post_layernorm: ClipLayerNorm,
    layers: Vec<ClipVisionTransformerBlock>,
}

impl ClipVisionTransformer {
    fn new(builder: &ClipModelBuilder, prefix: &str) -> anyhow::Result<Self> {
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
                    ClipVisionTransformerBlock::new(
                        builder,
                        &format!("{}.encoder.layers.{}", prefix, i),
                    )
                })
                .collect::<anyhow::Result<_>>()?,
            params: builder.params.vision_config.clone(),
        })
    }
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
        &self, hidden_state: &mut TensorViewMut<f16>, layer: &ClipVisionTransformerBlock,
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
    let builder = ClipModelBuilder::new(&model, &*kernels, &params);
    let vt = ClipVisionTransformer::new(&builder, "vision_model")?;
    let context = ClipVisionContext::new(Arc::new(vt), kernels);
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
