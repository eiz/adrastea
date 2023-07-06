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

use std::{fs::File, path::Path};

use alloc::sync::Arc;
use regex::Regex;
use sentencepiece::SentencePieceProcessor;
use serde::Deserialize;
use simt_hip::{HipDevice, HipPhysicalDevice};

use crate::{
    clip::{ClipModelLoader, ClipParams, ClipVisionContext, ClipVisionTransformer},
    kernels::{GpuKernels, MatmulTracer},
    llama::{HuggingFaceLlamaModelLoader, LlamaContext, LlamaModel, LlamaParams},
    pickle::{PickledModel, ShardedModel},
};

const IM_END: u32 = 32003;
const IM_PATCH: u32 = 32001;
const IM_START: u32 = 32002;

#[derive(Clone, Deserialize)]
pub struct LlavaParams {
    hidden_size: i32,
    mm_vision_select_layer: i32,
    mm_vision_tower: String,
    num_attention_heads: i32,
    num_hidden_layers: i32,
    vocab_size: i32,
    rms_norm_eps: f32,
}

impl LlavaParams {
    fn to_llama(&self) -> LlamaParams {
        LlamaParams {
            dim: self.hidden_size as u32,
            multiple_of: 256,
            n_heads: self.num_attention_heads as u32,
            n_layers: self.num_hidden_layers as u32,
            norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size as isize,
        }
    }
}

pub fn llava_test<P: AsRef<Path>, Q: AsRef<Path>, R: AsRef<Path>>(
    llava_path: P, clip_path: Q, images: &[R], prompt: &str,
) -> anyhow::Result<()> {
    let llava_path = llava_path.as_ref();
    let clip_path = clip_path.as_ref();
    let phys = HipPhysicalDevice::get(0)?;
    let device = Arc::new(HipDevice::new(phys)?);
    let _scope = device.lock()?;
    // BIG TODO: loading each kernel as a separate module like this is super not ergonomic
    // use a better way
    // lmao I'm still copypasta'ing this todo everywhere
    let kernels = Arc::new(MatmulTracer::new(GpuKernels::new(phys.capability()?)?));
    let clip_model = PickledModel::load_file(clip_path.join("pytorch_model.bin"), None)?;
    let clip_params: ClipParams =
        serde_json::from_reader(File::open(clip_path.join("config.json"))?)?;
    let clip_builder = ClipModelLoader::new(&clip_model, &*kernels, &clip_params);
    let clip_vision = ClipVisionTransformer::new(&clip_builder, "vision_model")?;
    let clip_vision = ClipVisionContext::new(Arc::new(clip_vision), kernels.clone());
    let llava_model = ShardedModel::load_huggingface(llava_path)?;
    let llava_params: LlavaParams =
        serde_json::from_reader(File::open(llava_path.join("config.json"))?)?;
    let llava_tokenizer = SentencePieceProcessor::open(llava_path.join("tokenizer.model"))?;
    let end_of_text = 1;
    let mut context = LlamaContext::new(
        Arc::new(LlamaModel::new(
            &HuggingFaceLlamaModelLoader::new(&llava_model, &llava_params.to_llama(), &*kernels),
            llava_params.to_llama(),
            llava_tokenizer,
            4,
        )?),
        kernels,
    );
    let image_subst_re = Regex::new(r#"\$\d"#).unwrap();
    let mut text_start = 0;
    let mut segments = vec![];

    #[derive(Debug)]
    enum PromptSeg {
        Text(String),
        Image(usize),
    }
    for rmatch in image_subst_re.find_iter(prompt) {
        if rmatch.start() > text_start {
            segments.push(PromptSeg::Text(prompt[text_start..rmatch.start()].to_string()));
        }
        segments.push(PromptSeg::Image(rmatch.as_str()[1..].parse()?));
        text_start = rmatch.end();
    }
    if text_start < prompt.len() {
        segments.push(PromptSeg::Text(prompt[text_start..].to_string()));
    }
    let mut token_buffer = vec![context.model().tokenizer().bos_id().unwrap() as i32];
    let mut patch_offsets = vec![];
    for seg in segments.iter() {
        match seg {
            PromptSeg::Text(text) => {
                let text = context.model().tokenizer().encode(text)?;
                for i in text {
                    token_buffer.push(i.id as i32);
                }
            }
            PromptSeg::Image(idx) => {
                token_buffer.push(IM_START as i32);
                patch_offsets.push((idx, token_buffer.len()));
                for _i in 0..clip_params.vision_config.image_size {
                    token_buffer.push(IM_PATCH as i32);
                }
                token_buffer.push(IM_END as i32);
            }
        }
    }
    println!("segments {:?}", segments);
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
            println!("end of text");
            break;
        }
        token_buffer.push(argmax as i32);
        println!(
            "text {:?}",
            context.model().tokenizer().decode_piece_ids(
                &token_buffer
                    .iter()
                    .filter(|&&x| x < context.model().tokenizer().len() as i32)
                    .map(|x| *x as u32)
                    .collect::<Vec<_>>()
            )
        );
    }
    todo!()
}
