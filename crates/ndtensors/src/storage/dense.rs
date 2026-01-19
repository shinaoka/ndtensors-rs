//! Dense storage for tensor data.
//!
//! This module provides the `Dense` storage type which holds tensor data
//! in a contiguous array. The storage is generic over a `DataBuffer` backend,
//! allowing for CPU (Vec<T>), GPU, or other implementations.

use std::marker::PhantomData;

use crate::scalar::Scalar;
use crate::storage::buffer::{CpuBuffer, DataBuffer};

/// Dense storage - contiguous array of elements in column-major order.
///
/// This mirrors NDTensors.jl's `Dense{ElT, DataT}` structure, where
/// the data buffer is generic to support different backends (CPU, GPU, etc.).
///
/// # Type Parameters
///
/// * `ElT` - Element type (e.g., f64, c64)
/// * `D` - Data buffer type, defaults to `CpuBuffer<ElT>`
#[derive(Debug, Clone, PartialEq)]
pub struct Dense<ElT: Scalar, D: DataBuffer<ElT> = CpuBuffer<ElT>> {
    data: D,
    _phantom: PhantomData<ElT>,
}

/// Type alias for CPU-backed dense storage.
pub type CpuDense<ElT> = Dense<ElT, CpuBuffer<ElT>>;

impl<ElT: Scalar, D: DataBuffer<ElT>> Dense<ElT, D> {
    /// Create dense storage with given length, zero-initialized.
    pub fn zeros(len: usize) -> Self {
        Self {
            data: D::zeros(len),
            _phantom: PhantomData,
        }
    }

    /// Create dense storage from existing vector (takes ownership).
    pub fn from_vec(data: Vec<ElT>) -> Self {
        Self {
            data: D::from_vec(data),
            _phantom: PhantomData,
        }
    }

    /// Create dense storage from an existing data buffer.
    pub fn from_buffer(data: D) -> Self {
        Self {
            data,
            _phantom: PhantomData,
        }
    }

    /// Length of storage.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if storage is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get immutable slice of data.
    #[inline]
    pub fn as_slice(&self) -> &[ElT] {
        self.data.as_slice()
    }

    /// Get mutable slice of data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [ElT] {
        self.data.as_mut_slice()
    }

    /// Get raw pointer (for FFI).
    #[inline]
    pub fn as_ptr(&self) -> *const ElT {
        self.data.as_ptr()
    }

    /// Get mutable raw pointer (for FFI).
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut ElT {
        self.data.as_mut_ptr()
    }

    /// Get a reference to the underlying data buffer.
    #[inline]
    pub fn buffer(&self) -> &D {
        &self.data
    }

    /// Get a mutable reference to the underlying data buffer.
    #[inline]
    pub fn buffer_mut(&mut self) -> &mut D {
        &mut self.data
    }

    /// Create a view of the same underlying data.
    ///
    /// This creates a new Dense that shares the same underlying storage
    /// using reference counting. The data is not copied.
    #[inline]
    pub fn view(&self) -> Self {
        Self {
            data: self.data.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<ElT: Scalar, D: DataBuffer<ElT>> std::ops::Index<usize> for Dense<ElT, D> {
    type Output = ElT;

    #[inline]
    fn index(&self, i: usize) -> &ElT {
        &self.data.as_slice()[i]
    }
}

impl<ElT: Scalar, D: DataBuffer<ElT>> std::ops::IndexMut<usize> for Dense<ElT, D> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut ElT {
        &mut self.data.as_mut_slice()[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let d: Dense<f64> = Dense::zeros(5);
        assert_eq!(d.len(), 5);
        assert!(!d.is_empty());
        for i in 0..5 {
            assert_eq!(d[i], 0.0);
        }
    }

    #[test]
    fn test_from_vec() {
        let d: Dense<f64> = Dense::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(d.len(), 3);
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 2.0);
        assert_eq!(d[2], 3.0);
    }

    #[test]
    fn test_index_mut() {
        let mut d: Dense<f64> = Dense::zeros(3);
        d[1] = 5.0;
        assert_eq!(d[1], 5.0);
    }

    #[test]
    fn test_as_slice() {
        let d: Dense<f64> = Dense::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(d.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_buffer() {
        let buf = CpuBuffer::from_vec(vec![1.0, 2.0, 3.0]);
        let d: Dense<f64> = Dense::from_buffer(buf);
        assert_eq!(d.len(), 3);
        assert_eq!(d[0], 1.0);
    }

    #[test]
    fn test_cpu_dense_alias() {
        let d: CpuDense<f64> = CpuDense::zeros(5);
        assert_eq!(d.len(), 5);
    }
}
