use std::{fs::File, io::Write, path::Path};

use alloc::sync::Arc;
use half::f16;
use memmap2::Mmap;
use simt_hip::{HipDevice, HipPhysicalDevice};
use skia_safe::{
    AlphaType, ColorSpace, ColorType, CubicResampler, Data, EncodedImageFormat, Image, ImageInfo,
    Pixmap, Surface,
};

use crate::{
    kernels::{CommonKernels, GpuKernels},
    pickle::{load_tensor, PickledModel},
    tensor::{Tensor, TensorLayout},
};

// this implements the 🤗 version of CLIP

fn to_f16(kernels: &dyn CommonKernels, tensor: Tensor<f32>) -> anyhow::Result<Tensor<f16>> {
    let mut output = Tensor::new_hip_layout(tensor.layout().clone())?;
    kernels.fp32_to_fp16(&mut output.as_view_mut(), &tensor.as_view())?;
    Ok(output)
}

struct ClipVisionTransformer {
    class_embedding: Tensor<f16>,
    patch_embedding: Tensor<f16>,
    position_embedding: Tensor<f16>,
}

impl ClipVisionTransformer {
    fn new(
        pickle: &PickledModel<()>, kernels: &dyn CommonKernels, prefix: &str,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            class_embedding: to_f16(
                kernels,
                load_tensor(pickle, &format!("{}.embeddings.class_embedding", prefix))?,
            )?,
            patch_embedding: to_f16(
                kernels,
                load_tensor(pickle, &format!("{}.embeddings.patch_embedding.weight", prefix))?,
            )?,
            position_embedding: to_f16(
                kernels,
                load_tensor(pickle, &format!("{}.embeddings.position_embedding.weight", prefix))?,
            )?,
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
    let model = PickledModel::load_file(path, None)?;
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
    // TODO: this does not give the same results as PIL =/
    canvas.draw_image_with_sampling_options(image, (0, 0), CubicResampler::catmull_rom(), None);
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
    let vt = ClipVisionTransformer::new(&model, &*kernels, "vision_model")?;
    let mut patch_embeds = Tensor::new_hip(&[1024, 16, 16])?;
    let zero_bias = Tensor::new_hip(&[1024])?;
    println!("values {:>7.4?}", values);
    println!("patch_embedding {:?}", vt.patch_embedding.layout());
    kernels.conv2d_f16(
        &mut patch_embeds.as_view_mut(),
        &values.as_view(),
        &vt.patch_embedding.as_view(),
        &zero_bias.as_view(),
        (14, 14),
        (0, 0),
        crate::kernels::Conv1dActivation::None,
    )?;
    println!("patch_embeds {:>7.4?}", patch_embeds);
    let v_patch_embeds = patch_embeds.as_view().shape_cast(&[1024, 256]).permute(&[1, 0]);
    todo!();
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use simt_hip::{HipDevice, HipPhysicalDevice};

    use crate::{
        kernels::{CommonKernels, GpuKernels},
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
        let expected_result: Tensor<f32> = load_tensor(&model, "clip.conv2d.result")?;
        let mut actual_result = Tensor::new_hip(&[1024, 16, 16])?;
        let mut err_stats_gpu = Tensor::new_hip(&[3])?;
        let bias = Tensor::new_hip(&[1024])?;
        kernels.conv2d_f32(
            &mut actual_result.as_view_mut(),
            &pixel_values.as_view(),
            &weight.as_view(),
            &bias.as_view(),
            (14, 14),
            (0, 0),
            crate::kernels::Conv1dActivation::None,
        )?;
        kernels.error_stats_f32(
            &mut err_stats_gpu.as_view_mut(),
            &actual_result.as_view(),
            &expected_result.as_view(),
        )?;
        let err_stats = err_stats_gpu.into_cpu()?;
        let err_stats = err_stats.storage().as_cpu();
        println!("expected result {:>7.4?}", expected_result);
        println!("actual result {:>7.4?}", actual_result);
        println!("err_stats {:?}", err_stats);
        assert!(err_stats[0] < 0.001);
        Ok(())
    }
}
