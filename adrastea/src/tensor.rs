use core::{
    fmt::{Debug, Display, Formatter},
    marker::PhantomData,
};

use simt_hip::HipBuffer;
use smallvec::SmallVec;

use crate::util::ElidingRangeIterator;

#[derive(Clone, Debug)]
pub struct TensorLayout {
    pub dims: SmallVec<[usize; 7]>,
    pub strides: SmallVec<[usize; 7]>,
}

impl TensorLayout {
    pub fn new(dims: &[usize], strides: &[usize]) -> Self {
        assert_eq!(dims.len(), strides.len());
        assert_ne!(dims.len(), 0);
        assert_ne!(strides.len(), 0);
        assert!(dims.iter().all(|&x| x != 0));
        assert!(strides.iter().all(|&x| x != 0));
        Self { dims: dims.into(), strides: strides.into() }
    }

    pub fn row_major(dims: &[usize]) -> Self {
        let mut strides = SmallVec::<[usize; 7]>::new();
        let mut stride = 1;
        for &dim in dims.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();
        Self::new(dims, &strides)
    }

    pub fn largest_address(&self) -> usize {
        let mut addr = 0;
        for (&dim, &stride) in self.dims.iter().zip(self.strides.iter()) {
            addr += (dim - 1) * stride;
        }
        addr
    }

    pub fn permute(&self, dim_order: &[usize]) -> Self {
        assert_eq!(dim_order.len(), self.dims.len());
        let mut dims = SmallVec::<[usize; 7]>::new();
        let mut strides = SmallVec::<[usize; 7]>::new();
        for &dim in dim_order.iter() {
            dims.push(self.dims[dim]);
            strides.push(self.strides[dim]);
        }
        Self::new(&dims, &strides)
    }

    // TODO: poor man's .view(), except it won't tell you if you mess up!
    pub fn shape_cast(&self, dims: &[isize]) -> Self {
        let mut mut_dims: SmallVec<[isize; 7]> = dims.into();
        let mut strides = SmallVec::<[usize; 7]>::new();
        let mut has_neg_dim = false;
        for dim in mut_dims.iter_mut() {
            if *dim < 0 {
                assert!(!has_neg_dim);
                has_neg_dim = true;
                *dim = self.dims.iter().product::<usize>() as isize
                    / dims.iter().filter(|&&x| x >= 0).product::<isize>();
            }
        }
        assert_eq!(
            mut_dims.iter().product::<isize>(),
            self.dims.iter().product::<usize>() as isize
        );
        let mut stride = 1;
        for &dim in mut_dims.iter().rev() {
            strides.push(stride as usize);
            stride *= dim;
        }
        strides.reverse();
        Self { dims: mut_dims.iter().map(|x| *x as usize).collect(), strides }
    }

    pub fn size(&self, dim: isize) -> usize {
        if dim < 0 {
            self.dims[self.dims.len() - (-dim as usize)]
        } else {
            self.dims[dim as usize]
        }
    }

    pub fn stride(&self, dim: isize) -> usize {
        if dim < 0 {
            self.strides[self.strides.len() - (-dim as usize)]
        } else {
            self.strides[dim as usize]
        }
    }

    pub fn skip(&self, sizes: &[isize]) -> (usize, Self) {
        let mut offset = 0;
        let mut new_dims = SmallVec::<[usize; 7]>::new();
        let mut new_strides = SmallVec::<[usize; 7]>::new();
        for ((dim, stride), size) in self.dims.iter().zip(self.strides.iter()).zip(sizes.iter()) {
            let size = if *size == -1 { *dim as isize } else { *size };
            assert!(size <= *dim as isize);
            assert!(size >= 0);
            offset += stride * size as usize;
            new_dims.push(dim - size as usize);
            new_strides.push(*stride);
        }
        (offset, Self::new(&new_dims, &new_strides))
    }

    pub fn take(&self, sizes: &[isize]) -> Self {
        let mut new_dims = SmallVec::<[usize; 7]>::new();
        let mut new_strides = SmallVec::<[usize; 7]>::new();
        for ((dim, stride), size) in self.dims.iter().zip(self.strides.iter()).zip(sizes.iter()) {
            let size = if *size == -1 { *dim as isize } else { *size };
            assert!(size <= *dim as isize);
            assert!(size >= 0);
            new_dims.push(size as usize);
            new_strides.push(*stride);
        }
        Self::new(&new_dims, &new_strides)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum TensorStoragePtr<T> {
    Cpu(*const T),
    Hip(simt_hip_sys::hipDeviceptr_t),
}

impl<T> Copy for TensorStoragePtr<T> {}
impl<T> Clone for TensorStoragePtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum TensorStoragePtrMut<T> {
    Cpu(*mut T),
    Hip(simt_hip_sys::hipDeviceptr_t),
}

impl<T> Copy for TensorStoragePtrMut<T> {}
impl<T> Clone for TensorStoragePtrMut<T> {
    fn clone(&self) -> Self {
        *self
    }
}

pub enum TensorStorage<T> {
    Cpu(Vec<T>),
    Hip(HipBuffer),
}

impl<T> TensorStorage<T> {
    pub fn as_cpu(&self) -> &Vec<T> {
        match self {
            TensorStorage::Cpu(v) => v,
            _ => panic!("tensor storage not resident on cpu"),
        }
    }

    pub fn as_mut_cpu(&mut self) -> &mut Vec<T> {
        match self {
            TensorStorage::Cpu(v) => v,
            _ => panic!("tensor storage not resident on cpu"),
        }
    }

    pub fn as_hip(&self) -> &HipBuffer {
        match self {
            TensorStorage::Hip(b) => b,
            _ => panic!("tensor storage not resident on gpu"),
        }
    }

    pub fn as_mut_hip(&mut self) -> &mut HipBuffer {
        match self {
            TensorStorage::Hip(b) => b,
            _ => panic!("tensor storage not resident on gpu"),
        }
    }
}

pub struct Tensor<T> {
    storage: TensorStorage<T>,
    layout: TensorLayout,
    _dead: PhantomData<T>,
}

impl<T> Tensor<T> {
    pub fn as_view(&self) -> TensorView<T> {
        TensorView {
            ptr: match &self.storage {
                TensorStorage::Cpu(v) => TensorStoragePtr::Cpu(v.as_ptr()),
                TensorStorage::Hip(b) => TensorStoragePtr::Hip(b.ptr),
            },
            layout: self.layout.clone(),
            _dead: PhantomData,
        }
    }

    pub fn as_view_mut(&mut self) -> TensorViewMut<T> {
        TensorViewMut {
            ptr: match &mut self.storage {
                TensorStorage::Cpu(v) => TensorStoragePtrMut::Cpu(v.as_mut_ptr()),
                TensorStorage::Hip(b) => TensorStoragePtrMut::Hip(b.ptr),
            },
            layout: self.layout.clone(),
            _dead: PhantomData,
        }
    }
}

impl<T: Copy + Default> Tensor<T> {
    pub fn new_cpu(dims: &[usize]) -> Self {
        let layout = TensorLayout::row_major(dims);
        let storage = TensorStorage::Cpu(vec![T::default(); layout.largest_address() + 1]);
        Tensor { storage, layout, _dead: PhantomData }
    }

    pub fn new_hip(dims: &[usize]) -> anyhow::Result<Self> {
        Self::new_hip_layout(TensorLayout::row_major(dims))
    }

    pub fn new_hip_layout(layout: TensorLayout) -> anyhow::Result<Self> {
        let buf = HipBuffer::new((layout.largest_address() + 1) * std::mem::size_of::<T>())?;
        unsafe {
            simt_hip::hip_call(|| {
                simt_hip_sys::library().hipMemset(buf.ptr as *mut _, 0, buf.size)
            })?;
        }
        Ok(Tensor { storage: TensorStorage::Hip(buf), layout, _dead: PhantomData })
    }

    pub fn from_vec(vec: Vec<T>, layout: TensorLayout) -> Self {
        assert!(vec.len() > layout.largest_address());
        Tensor { storage: TensorStorage::Cpu(vec), layout, _dead: PhantomData }
    }

    pub fn into_hip(self) -> anyhow::Result<Self> {
        match self.storage {
            TensorStorage::Cpu(v) => {
                let mut buf = HipBuffer::new(v.len() * std::mem::size_of::<T>())?;
                buf.copy_from_slice(&v)?;
                Ok(Tensor {
                    storage: TensorStorage::Hip(buf),
                    layout: self.layout,
                    _dead: PhantomData,
                })
            }
            TensorStorage::Hip(_) => Ok(self),
        }
    }

    pub fn into_cpu(self) -> anyhow::Result<Self> {
        match self.storage {
            TensorStorage::Cpu(_) => Ok(self),
            TensorStorage::Hip(b) => {
                let mut v = vec![T::default(); b.size / std::mem::size_of::<T>()];
                b.copy_to_slice(&mut v)?;
                Ok(Tensor {
                    storage: TensorStorage::Cpu(v),
                    layout: self.layout,
                    _dead: PhantomData,
                })
            }
        }
    }

    pub fn as_gpu_ptr(&self) -> *const T {
        match &self.storage {
            TensorStorage::Cpu(_) => panic!("not a gpu tensor"),
            TensorStorage::Hip(b) => b.ptr as *const T,
        }
    }

    pub fn as_mut_gpu_ptr(&mut self) -> *mut T {
        match &self.storage {
            TensorStorage::Cpu(_) => panic!("not a gpu tensor"),
            TensorStorage::Hip(b) => b.ptr as *mut T,
        }
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    pub fn storage(&self) -> &TensorStorage<T> {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut TensorStorage<T> {
        &mut self.storage
    }

    pub fn size(&self, dim: isize) -> usize {
        self.layout.size(dim)
    }

    pub fn stride(&self, dim: isize) -> usize {
        self.layout.stride(dim)
    }
}

impl<T: Default + Debug + Copy> Debug for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(f)
    }
}

#[repr(C)]
pub struct TensorView<'a, T> {
    ptr: TensorStoragePtr<T>,
    layout: TensorLayout,
    _dead: PhantomData<&'a T>,
}

fn format_slice_with_layout<T: Debug + Copy>(
    f: &mut Formatter<'_>,
    slice: &[T],
    dim: usize,
    layout: &TensorLayout,
) -> std::fmt::Result {
    let dims_right = layout.dims.len() - dim - 1;
    let mut first = true;
    if dims_right == 0 {
        for (skip, i) in ElidingRangeIterator::new(layout.dims[dim], 6, 3) {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            Debug::fmt(&slice[layout.strides[dim] * i], f)?;
            if skip {
                f.write_str(", ...")?;
            }
        }
    } else {
        for (skip, i) in ElidingRangeIterator::new(layout.dims[dim], 6, 3) {
            if !first {
                write!(f, "\n")?;
            }
            first = false;
            write!(f, "[")?;
            format_slice_with_layout(f, &slice[layout.strides[dim] * i..], dim + 1, layout)?;
            write!(f, "]")?;
            if skip {
                write!(f, "\n...")?;
            }
        }
    }
    Ok(())
}

impl<'a, T: Default + Debug + Copy> Debug for TensorView<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.ptr {
            TensorStoragePtr::Cpu(p) => {
                let slice =
                    unsafe { std::slice::from_raw_parts(p, self.layout.largest_address() + 1) };
                format_slice_with_layout(f, slice, 0, &self.layout)?;
            }
            TensorStoragePtr::Hip(b) => {
                let storage_size = self.layout.largest_address() + 1;
                let mut cpu_data = vec![T::default(); storage_size];
                unsafe {
                    simt_hip::hip_call(|| {
                        simt_hip_sys::library().hipMemcpy(
                            cpu_data.as_mut_ptr() as *mut std::ffi::c_void,
                            b as *const std::ffi::c_void,
                            storage_size * std::mem::size_of::<T>(),
                            simt_hip_sys::hipMemcpyKind::hipMemcpyDeviceToHost,
                        )
                    })
                    .map_err(|_| std::fmt::Error)?;
                }
                format_slice_with_layout(f, &cpu_data, 0, &self.layout)?;
            }
        }
        Ok(())
    }
}

impl<'a, T> TensorView<'a, T> {
    pub fn as_gpu_ptr(&self) -> *const T {
        match self.ptr {
            TensorStoragePtr::Cpu(_) => panic!("not a gpu tensor"),
            TensorStoragePtr::Hip(b) => b as *const T,
        }
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    pub fn size(&self, dim: isize) -> usize {
        self.layout.size(dim)
    }

    pub fn stride(&self, dim: isize) -> usize {
        self.layout.stride(dim)
    }

    pub fn permute(&self, dim_order: &[usize]) -> Self {
        Self { ptr: self.ptr, layout: self.layout.permute(dim_order), _dead: PhantomData }
    }

    pub fn shape_cast(&self, shape: &[isize]) -> Self {
        Self { ptr: self.ptr, layout: self.layout.shape_cast(shape), _dead: PhantomData }
    }
}

#[repr(C)]
pub struct TensorViewMut<'a, T> {
    ptr: TensorStoragePtrMut<T>,
    layout: TensorLayout,
    _dead: PhantomData<&'a mut T>,
}

impl<'a, T> TensorViewMut<'a, T> {
    pub fn as_view(&self) -> TensorView<'_, T> {
        TensorView {
            ptr: match &self.ptr {
                TensorStoragePtrMut::Cpu(p) => TensorStoragePtr::Cpu(*p),
                TensorStoragePtrMut::Hip(p) => TensorStoragePtr::Hip(*p),
            },
            layout: self.layout.clone(),
            _dead: PhantomData,
        }
    }

    pub fn as_gpu_ptr(&self) -> *const T {
        match self.ptr {
            TensorStoragePtrMut::Cpu(_) => panic!("not a gpu tensor"),
            TensorStoragePtrMut::Hip(b) => b as *const T,
        }
    }

    pub fn as_mut_gpu_ptr(&mut self) -> *mut T {
        match self.ptr {
            TensorStoragePtrMut::Cpu(_) => panic!("not a gpu tensor"),
            TensorStoragePtrMut::Hip(b) => b as *mut T,
        }
    }

    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    pub fn size(&self, dim: isize) -> usize {
        self.layout.size(dim)
    }

    pub fn stride(&self, dim: isize) -> usize {
        self.layout.stride(dim)
    }

    pub fn permute(&mut self, dim_order: &[usize]) -> Self {
        Self { ptr: self.ptr, layout: self.layout.permute(dim_order), _dead: PhantomData }
    }

    pub fn shape_cast(&mut self, shape: &[isize]) -> Self {
        Self { ptr: self.ptr, layout: self.layout.shape_cast(shape), _dead: PhantomData }
    }
}

impl<'a, T: Display + Default + Debug + Copy> Debug for TensorViewMut<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(f)
    }
}
