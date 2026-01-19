//! Data buffer abstraction for backend-agnostic storage.
//!
//! This module provides the `DataBuffer` trait which mirrors NDTensors.jl's
//! `DataT` type parameter in `Dense{ElT, DataT}`. Different implementations
//! allow for CPU (Vec<T>), GPU (future: CudaBuffer), or other backends.

use crate::scalar::Scalar;

/// Trait for backend-agnostic data storage.
///
/// This mirrors NDTensors.jl's `DataT` in `Dense{ElT, DataT}`.
/// Different implementations:
/// - `CpuBuffer<T>` - wraps `Vec<T>` for CPU computation
/// - Future: `CudaBuffer<T>` for CUDA GPU
/// - Future: `MetalBuffer<T>` for Apple Metal GPU
///
/// # Design Notes
///
/// Operations should be generic over `DataBuffer` trait, not hardcoded to `Vec<T>`.
/// Backend-specific optimizations (e.g., faer for CPU) go in specialized impls.
pub trait DataBuffer<T: Scalar>: Clone + std::fmt::Debug + PartialEq {
    /// Create buffer with given length, zero-initialized.
    fn zeros(len: usize) -> Self;

    /// Create buffer from existing vector.
    fn from_vec(data: Vec<T>) -> Self;

    /// Length of buffer (number of elements).
    fn len(&self) -> usize;

    /// Check if buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get immutable slice of data.
    ///
    /// # Note
    /// This method requires CPU-accessible memory. For GPU buffers,
    /// this may involve a device-to-host copy.
    fn as_slice(&self) -> &[T];

    /// Get mutable slice of data.
    ///
    /// # Note
    /// This method requires CPU-accessible memory. For GPU buffers,
    /// this may involve a device-to-host copy and subsequent sync.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Get raw pointer (for FFI).
    fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    /// Get mutable raw pointer (for FFI).
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
    }
}

/// CPU data buffer backed by `Vec<T>`.
///
/// This is the default backend for dense tensor storage, using standard
/// Rust vectors for data storage. It provides zero-copy access to the
/// underlying data and is compatible with faer's matrix operations.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuBuffer<T: Scalar> {
    data: Vec<T>,
}

impl<T: Scalar> CpuBuffer<T> {
    /// Create a new CpuBuffer from a Vec.
    #[inline]
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Get a reference to the underlying Vec.
    #[inline]
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Get a mutable reference to the underlying Vec.
    #[inline]
    pub fn data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// Consume the buffer and return the underlying Vec.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Scalar> DataBuffer<T> for CpuBuffer<T> {
    #[inline]
    fn zeros(len: usize) -> Self {
        Self {
            data: vec![T::zero(); len],
        }
    }

    #[inline]
    fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Scalar> std::ops::Index<usize> for CpuBuffer<T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T: Scalar> std::ops::IndexMut<usize> for CpuBuffer<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_buffer_zeros() {
        let buf: CpuBuffer<f64> = CpuBuffer::zeros(5);
        assert_eq!(buf.len(), 5);
        assert!(!buf.is_empty());
        for i in 0..5 {
            assert_eq!(buf[i], 0.0);
        }
    }

    #[test]
    fn test_cpu_buffer_from_vec() {
        let buf = CpuBuffer::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 1.0);
        assert_eq!(buf[1], 2.0);
        assert_eq!(buf[2], 3.0);
    }

    #[test]
    fn test_cpu_buffer_as_slice() {
        let buf = CpuBuffer::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(buf.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cpu_buffer_as_mut_slice() {
        let mut buf = CpuBuffer::from_vec(vec![1.0, 2.0, 3.0]);
        buf.as_mut_slice()[1] = 5.0;
        assert_eq!(buf[1], 5.0);
    }

    #[test]
    fn test_cpu_buffer_index_mut() {
        let mut buf: CpuBuffer<f64> = CpuBuffer::zeros(3);
        buf[1] = 5.0;
        assert_eq!(buf[1], 5.0);
    }

    #[test]
    fn test_cpu_buffer_clone() {
        let buf1 = CpuBuffer::from_vec(vec![1.0, 2.0, 3.0]);
        let buf2 = buf1.clone();
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn test_cpu_buffer_into_vec() {
        let buf = CpuBuffer::from_vec(vec![1.0, 2.0, 3.0]);
        let vec = buf.into_vec();
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cpu_buffer_empty() {
        let buf: CpuBuffer<f64> = CpuBuffer::zeros(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }
}
