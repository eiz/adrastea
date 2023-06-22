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

use core::ops::Range;
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::Cursor,
    path::Path,
};

use anyhow::bail;
use half::f16;
use memmap2::Mmap;
use serde::Deserialize;
use zip::{CompressionMethod, ZipArchive};

use crate::tensor::{Tensor, TensorLayout, TensorStorage};

// ya ya ya bla bla undefined behavior. whatever. make sure nobody secretly boops your files
// while they are in use
pub struct MappedBuffer {
    _fp: File,
    mapping: Mmap,
}

impl MappedBuffer {
    pub fn open<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let fp = File::open(path)?;
        let mapping = unsafe { Mmap::map(&fp)? };
        Ok(Self { _fp: fp, mapping })
    }

    pub fn data(&self) -> &[u8] {
        &self.mapping
    }
}

fn value_at_path<'a>(
    mut value: &'a serde_pickle::Value, path: &[&str],
) -> anyhow::Result<&'a serde_pickle::Value> {
    let mut i = 0;
    loop {
        if i < path.len() {
            let dict = match value {
                serde_pickle::Value::Dict(map) => map,
                _ => bail!("expected dict"),
            };

            // TODO: really ugly
            value = match dict.get(&serde_pickle::HashableValue::String(path[i].to_string())) {
                Some(value) => value,
                None => bail!("key not found"),
            };
            i += 1;
        } else {
            return Ok(value);
        }
    }
}

fn value_as_dict<'a>(
    value: &'a serde_pickle::Value,
) -> anyhow::Result<&'a BTreeMap<serde_pickle::HashableValue, serde_pickle::Value>> {
    match value_at_path(value, &[])? {
        serde_pickle::Value::Dict(map) => Ok(map),
        _ => bail!("expected dict"),
    }
}

pub trait ModelState<'de>: Deserialize<'de> {
    type Metadata;
    type LoadParams;
    fn init(&mut self, _params: Self::LoadParams) {}
    fn state_dict(&self) -> &serde_pickle::Value;
    fn into_metadata(self) -> Self::Metadata;
    fn load<P: AsRef<Path>>(
        path: P, params: Self::LoadParams,
    ) -> anyhow::Result<PickledModel<Self::Metadata>> {
        PickledModel::load_typed::<Self, P>(path, params)
    }
}

type PyTensorStorage = (String, String, String, String, i64);
// TODO this is probably gonna break when new fields are added (e.g. 'metadata' already exists now)
type PyTensor = (PyTensorStorage, i64, Vec<i64>, Vec<i64>, bool, serde_pickle::Value);

#[derive(Clone, Copy, Eq, Debug, PartialEq)]
pub enum TensorDataType {
    I8,
    F16,
    F32,
    F64,
}

impl TensorDataType {
    pub fn element_size(&self) -> usize {
        match self {
            TensorDataType::I8 => 1,
            TensorDataType::F16 => 2,
            TensorDataType::F32 => 4,
            TensorDataType::F64 => 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PickledTensor {
    pub dtype: TensorDataType,
    pub shape: [usize; 4],
    pub stride: [usize; 4],
    pub range: Range<usize>,
}

fn expand_dims_4d(dims: &[usize]) -> anyhow::Result<[usize; 4]> {
    let result = match dims.len() {
        0 => bail!("no dimensions"),
        1 => [1, 1, 1, dims[0]],
        2 => [1, 1, dims[0], dims[1]],
        3 => [1, dims[0], dims[1], dims[2]],
        4 => [dims[0], dims[1], dims[2], dims[3]],
        _ => bail!("too many dimensions"),
    };
    Ok(result)
}

fn expand_stride_4d(dims: &[usize], strides: &[usize]) -> anyhow::Result<[usize; 4]> {
    let dims_product = dims.iter().product::<usize>();
    let result = match strides.len() {
        0 => bail!("no dimensions"),
        1 => [dims_product, dims_product, dims_product, strides[0]],
        2 => [dims_product, dims_product, strides[0], strides[1]],
        3 => [dims_product, strides[0], strides[1], strides[2]],
        4 => [strides[0], strides[1], strides[2], strides[3]],
        _ => bail!("too many dimensions"),
    };
    Ok(result)
}

fn tensors_from_dict(
    dict: &BTreeMap<serde_pickle::HashableValue, serde_pickle::Value>,
    filehash: HashMap<String, Range<usize>>,
) -> Result<HashMap<String, PickledTensor>, anyhow::Error> {
    let mut tensorhash = HashMap::new();
    for (k, v) in dict.iter() {
        if let serde_pickle::HashableValue::String(ref s) = k {
            if s != "_metadata" {
                let v: PyTensor = serde_pickle::from_value::<PyTensor>(v.clone())?;
                // TODO: storage offset
                let (storage, _storage_offset, size, stride, _requires_grad, _backwards_hooks) = v;
                let (_storage, dtype, data_idx, _location, _size) = storage;
                let tensor_data = filehash
                    .get(&data_idx)
                    .ok_or_else(|| anyhow::anyhow!("tensor data not found: {}", data_idx))?;
                let size = size.iter().map(|x| *x as usize).collect::<Vec<_>>();
                let stride = stride.iter().map(|x| *x as usize).collect::<Vec<_>>();
                let size4d = expand_dims_4d(&size)?;
                let stride4d = expand_stride_4d(&size, &stride)?;
                tensorhash.insert(
                    s.clone(),
                    PickledTensor {
                        dtype: match dtype.as_str() {
                            "half" => TensorDataType::F16,
                            _ => bail!("unsupported dtype: {}", dtype),
                        },
                        shape: size4d,
                        stride: stride4d,
                        range: tensor_data.clone(),
                    },
                );
            }
        }
    }
    Ok(tensorhash)
}

pub struct PickledModel<T> {
    pub mapping: MappedBuffer,
    pub tensors: HashMap<String, PickledTensor>,
    pub metadata: T,
}

struct RawModel<'a>(Option<&'a str>, serde_pickle::Value);

impl<'a, 'de> ModelState<'de> for RawModel<'a> {
    type Metadata = ();
    type LoadParams = Option<&'a str>;
    fn init(&mut self, params: Self::LoadParams) {
        self.0 = params;
    }
    fn state_dict(&self) -> &serde_pickle::Value {
        match self.0 {
            // TODO give this an error return
            Some(name) => value_at_path(&self.1, &[name]).unwrap(),
            None => &self.1,
        }
    }
    fn into_metadata(self) -> Self::Metadata {}
}

impl<'a, 'de> Deserialize<'de> for RawModel<'a> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_pickle::Value::deserialize(deserializer)?;
        Ok(RawModel(None, value))
    }
}

impl PickledModel<()> {
    pub fn load_typed<'de, T: ModelState<'de>, P: AsRef<Path>>(
        path: P, params: T::LoadParams,
    ) -> anyhow::Result<PickledModel<T::Metadata>> {
        let path = path.as_ref();
        let buf = MappedBuffer::open(path)?;
        let mut archive = ZipArchive::new(Cursor::new(buf.data()))?;
        let mut filehash = HashMap::new();
        for i in 0..archive.len() {
            let entry = archive.by_index(i)?;
            if entry.compression() != CompressionMethod::Stored {
                bail!("pytorch archives should not contain compressed files");
            }
            let path = Path::new(entry.name());
            if let Some(name) = path.file_name() {
                filehash.insert(
                    name.to_string_lossy().to_string(),
                    entry.data_start() as usize..(entry.data_start() + entry.size()) as usize,
                );
            }
        }
        let content_range =
            filehash.get("data.pkl").ok_or_else(|| anyhow::anyhow!("data.pkl not found"))?;
        let mut model: T = serde_pickle::from_slice(
            &buf.data()[content_range.clone()],
            serde_pickle::DeOptions::new(),
        )?;
        model.init(params);
        let dict = value_as_dict(model.state_dict())?;
        let tensors = tensors_from_dict(dict, filehash)?;
        Ok(PickledModel { mapping: buf, tensors, metadata: model.into_metadata() })
    }

    pub fn load_file<P: AsRef<Path>>(
        path: P, dict_path: Option<&str>,
    ) -> anyhow::Result<PickledModel<()>> {
        Self::load_typed::<RawModel, _>(path, dict_path)
    }
}

pub fn load_tensor<T>(pickled: &PickledModel<T>, name: &str) -> anyhow::Result<Tensor<f16>> {
    let pickled_tensor =
        pickled.tensors.get(name).ok_or_else(|| anyhow::anyhow!("tensor {} not found", name))?;
    let mut tensor =
        Tensor::new_hip_layout(TensorLayout::new(&pickled_tensor.shape, &pickled_tensor.stride))?;
    match tensor.storage_mut() {
        TensorStorage::Hip(ref mut b) => {
            b.copy_from_slice(&pickled.mapping.data()[pickled_tensor.range.clone()])?;
        }
        _ => unreachable!(),
    }
    Ok(tensor)
}
