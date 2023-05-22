use core::ops::Range;
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::Cursor,
    path::Path,
};

use anyhow::bail;
use memmap2::Mmap;
use zip::{CompressionMethod, ZipArchive};

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

fn value_as_dict(
    mut value: serde_pickle::Value,
    path: &[&str],
) -> anyhow::Result<BTreeMap<serde_pickle::HashableValue, serde_pickle::Value>> {
    let mut i = 0;
    loop {
        let dict = match value {
            serde_pickle::Value::Dict(map) => map,
            _ => bail!("expected dict"),
        };

        if i < path.len() {
            // TODO: really ugly
            value = match dict.get(&serde_pickle::HashableValue::String(path[i].to_string())) {
                Some(value) => value.clone(),
                None => bail!("key not found"),
            };
            i += 1;
        } else {
            return Ok(dict);
        }
    }
}

type PyTensorStorage = (String, String, String, String, i64);
// TODO this is probably gonna break when new fields are added (e.g. 'metadata' already exists now)
type PyTensor = (
    PyTensorStorage,
    i64,
    Vec<i64>,
    Vec<i64>,
    bool,
    serde_pickle::Value,
);

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
    pub shape: (i64, i64, i64, i64),
    pub stride: (i64, i64, i64, i64),
    pub range: Range<usize>,
}

fn expand_dims_4d(dims: &[i64]) -> anyhow::Result<(i64, i64, i64, i64)> {
    let result = match dims.len() {
        0 => bail!("no dimensions"),
        1 => (1, 1, 1, dims[0]),
        2 => (1, 1, dims[0], dims[1]),
        3 => (1, dims[0], dims[1], dims[2]),
        4 => (dims[0], dims[1], dims[2], dims[3]),
        _ => bail!("too many dimensions"),
    };
    Ok(result)
}

fn expand_stride_4d(dims: &[i64], strides: &[i64]) -> anyhow::Result<(i64, i64, i64, i64)> {
    let dims_product = dims.iter().product::<i64>();
    let result = match strides.len() {
        0 => bail!("no dimensions"),
        1 => (dims_product, dims_product, dims_product, strides[0]),
        2 => (dims_product, dims_product, strides[0], strides[1]),
        3 => (dims_product, strides[0], strides[1], strides[2]),
        4 => (strides[0], strides[1], strides[2], strides[3]),
        _ => bail!("too many dimensions"),
    };
    Ok(result)
}

fn tensors_from_dict(
    dict: BTreeMap<serde_pickle::HashableValue, serde_pickle::Value>,
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

pub struct PickledModel {
    pub mapping: MappedBuffer,
    pub tensors: HashMap<String, PickledTensor>,
}

impl PickledModel {
    pub fn load_file<P: AsRef<Path>>(path: P, dict_path: Option<&str>) -> anyhow::Result<Self> {
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
        let content_range = filehash
            .get("data.pkl")
            .ok_or_else(|| anyhow::anyhow!("data.pkl not found"))?;
        let pickle = serde_pickle::value_from_slice(
            &buf.data()[content_range.clone()],
            serde_pickle::DeOptions::new(),
        )?;
        let dict = match dict_path {
            Some(dict_path) => value_as_dict(pickle, &[dict_path]),
            _ => value_as_dict(pickle, &[]),
        }?;
        let tensors = tensors_from_dict(dict, filehash)?;
        Ok(PickledModel {
            mapping: buf,
            tensors,
        })
    }
}
