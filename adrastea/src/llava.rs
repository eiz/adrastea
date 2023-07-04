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

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use alloc::sync::Arc;
use sentencepiece::SentencePieceProcessor;
use serde::Deserialize;
use simt_hip::{HipDevice, HipPhysicalDevice};

use crate::{
    kernels::{GpuKernels, MatmulTracer},
    llama::{HuggingFaceLlamaModelLoader, LlamaContext, LlamaModel, LlamaParams},
    pickle::ShardedModel,
};

#[derive(Deserialize)]
pub struct LlavaParams {
    hidden_size: i32,
    mm_vision_select_layer: i32,
    mm_vision_tower: String,
    num_attention_heads: i32,
    num_hidden_layers: i32,
    vocab_size: i32,
}

pub fn llava_test() -> anyhow::Result<()> {
    let phys = HipPhysicalDevice::get(0)?;
    let device = Arc::new(HipDevice::new(phys)?);
    let _scope = device.lock()?;
    // BIG TODO: loading each kernel as a separate module like this is super not ergonomic
    // use a better way
    // lmao I'm still copypasta'ing this todo everywhere
    let kernels = Arc::new(MatmulTracer::new(GpuKernels::new(phys.capability()?)?));
    let path = Path::new("/home/eiz/Downloads/llava-7b");
    let model = ShardedModel::load_huggingface(&path)?;
    // TODO: copied the meta params.json into llava dir
    let params: LlamaParams = serde_json::from_reader(File::open(path.join("params.json"))?)?;
    let tokenizer = SentencePieceProcessor::open(path.join("tokenizer.model"))?;
    let end_of_text = 1;
    let mut context = LlamaContext::new(
        Arc::new(LlamaModel::new(&HuggingFaceLlamaModelLoader::new(model), params, tokenizer, 4)?),
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
        println!(
            "text {:?}",
            context
                .model()
                .tokenizer()
                .decode_piece_ids(&token_buffer.iter().map(|x| *x as u32).collect::<Vec<_>>())
        );
        token_buffer.push(argmax as i32);
    }
    todo!()
}
