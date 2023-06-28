use std::{fs::File, io::Write, path::Path};

use alloc::sync::Arc;
use half::f16;
use memmap2::Mmap;
use serde::Deserialize;
use simt_hip::{HipDevice, HipPhysicalDevice};
use skia_safe::{
    AlphaType, ColorSpace, ColorType, CubicResampler, Data, EncodedImageFormat, Image, ImageInfo,
    MipmapMode, Pixmap, SamplingOptions, Surface,
};

use crate::{
    kernels::{BinaryOp, CommonKernels, GpuKernels, UnaryOp},
    pickle::{load_tensor, PickledModel},
    tensor::{Tensor, TensorLayout},
};

// this implements the 🤗 version of CLIP

fn to_f16(kernels: &dyn CommonKernels, tensor: Tensor<f32>) -> anyhow::Result<Tensor<f16>> {
    let mut output = Tensor::new_hip_layout(tensor.layout().clone())?;
    kernels.fp32_to_fp16(&mut output.as_view_mut(), &tensor.as_view())?;
    Ok(output)
}

#[derive(Debug, Deserialize)]
struct ClipVisionParams {
    image_size: i32,
    patch_size: i32,
    num_attention_heads: i32,
    num_hidden_layers: i32,
}

#[derive(Debug, Deserialize)]
struct ClipTextParams {
    num_attention_heads: i32,
    num_hidden_layers: i32,
    vocab_size: i32,
}

#[derive(Debug, Deserialize)]
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
    query: Tensor<f16>,
    key: Tensor<f16>,
    value: Tensor<f16>,
    out: Tensor<f16>,
}

impl ClipAttention {
    pub fn new(builder: &ClipModelBuilder, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            query: builder.load_tensor_f16(&format!("{}.q_proj.weight", prefix))?,
            key: builder.load_tensor_f16(&format!("{}.k_proj.weight", prefix))?,
            value: builder.load_tensor_f16(&format!("{}.v_proj.weight", prefix))?,
            out: builder.load_tensor_f16(&format!("{}.out_proj.weight", prefix))?,
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

struct ClipFeedForward {
    fc1: ClipLinear,
    fc2: ClipLinear,
}

impl ClipFeedForward {
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
    mlp: ClipFeedForward,
}

impl ClipVisionTransformerBlock {
    pub fn new(builder: &ClipModelBuilder, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            layer_norm1: ClipLayerNorm::new(builder, &format!("{}.layer_norm1", prefix))?,
            attn: ClipAttention::new(builder, &format!("{}.self_attn", prefix))?,
            layer_norm2: ClipLayerNorm::new(builder, &format!("{}.layer_norm2", prefix))?,
            mlp: ClipFeedForward::new(builder, &format!("{}.mlp", prefix))?,
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

pub fn clip_test<P: AsRef<Path>>(path: P) -> anyhow::Result<()> {
    let path = path.as_ref();
    let phys = HipPhysicalDevice::get(0)?;
    let device = Arc::new(HipDevice::new(phys)?);
    let _scope = device.lock()?;
    let kernels = Arc::new(GpuKernels::new(phys.capability()?)?);
    let model = PickledModel::load_file(path.join("pytorch_model.bin"), None)?;
    let params: ClipParams = serde_json::from_reader(File::open(path.join("config.json"))?)?;
    let image = load_image_eager("/home/eiz/clip_test.jpg")?;
    let pixdims = image.dimensions();
    let (new_height, new_width) = resize_dims(pixdims.height, pixdims.width, 224);
    let mut surface = Surface::new_raster(
        &ImageInfo::new((224, 224), ColorType::BGRA8888, AlphaType::Premul, ColorSpace::new_srgb()),
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
    let mut sample_options: SamplingOptions = CubicResampler::catmull_rom().into();
    sample_options.mipmap = MipmapMode::Linear;
    println!("{:?}", sample_options);
    // TODO: this does not give the same results as PIL =/
    canvas.draw_image_with_sampling_options(image, (0, 0), sample_options, None);
    let pixmap = surface.peek_pixels().unwrap();
    let values = to_f16(
        &*kernels,
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
    let builder = ClipModelBuilder::new(&model, &*kernels, &params);
    let vt = ClipVisionTransformer::new(&builder, "vision_model")?;
    let mut embedding = Tensor::new_hip(&[257, 1024])?;
    let mut class_embed = embedding.as_view_mut();
    let mut class_embed = class_embed.take(&[1, 1024]);
    kernels.elementwise_unary_2d_f16(
        &mut class_embed,
        &vt.class_embedding.as_view().shape_cast(&[1, 1024]),
        UnaryOp::Identity,
    )?;
    let mut patch_embeds = embedding.as_view_mut();
    let mut patch_embeds =
        patch_embeds.skip(&[1, 0]).shape_cast(&[16, 16, 1024]).permute(&[2, 0, 1]);
    let zero_bias = Tensor::new_hip(&[1024])?;
    kernels.conv2d_f16(
        &mut patch_embeds,
        &values.as_view(),
        &vt.patch_embedding.as_view(),
        &zero_bias.as_view(),
        (14, 14),
        (0, 0),
        crate::kernels::Conv1dActivation::None,
    )?;
    kernels.elementwise_binary_2d_f16_inplace(
        &mut embedding.as_view_mut(),
        &vt.position_embedding.as_view(),
        BinaryOp::Add,
    )?;
    let mut hidden_state = Tensor::new_hip(&[257, 1024])?;
    kernels.layer_norm(
        &mut hidden_state.as_view_mut(),
        &embedding.as_view(),
        &vt.pre_layernorm.weight.as_view(),
        &vt.pre_layernorm.bias.as_view(),
        1.0e-5,
    )?;
    println!("initial hidden state {:>7.4?}", hidden_state);
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
