//! Data buffer abstraction for backend-agnostic storage.
//!
//! This module provides the `DataBuffer` trait which mirrors NDTensors.jl's
//! `DataT` type parameter in `Dense{ElT, DataT}`. Different implementations
//! allow for CPU (Vec<T>), GPU (future: CudaBuffer), or other backends.

use std::rc::Rc;

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

/// CPU data buffer backed by `Rc<Vec<T>>`.
///
/// This is the default backend for dense tensor storage, using standard
/// Rust vectors for data storage. It provides zero-copy access to the
/// underlying data and is compatible with faer's matrix operations.
///
/// The use of `Rc` enables zero-copy views (reshape) through shared ownership.
/// Mutation triggers copy-on-write semantics via `Rc::make_mut`.
#[derive(Debug, Clone)]
pub struct CpuBuffer<T: Scalar> {
    data: Rc<Vec<T>>,
}

impl<T: Scalar> PartialEq for CpuBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        // Compare by data content, not Rc identity
        self.data.as_ref() == other.data.as_ref()
    }
}

impl<T: Scalar> CpuBuffer<T> {
    /// Create a new CpuBuffer from a Vec.
    #[inline]
    pub fn new(data: Vec<T>) -> Self {
        Self {
            data: Rc::new(data),
        }
    }

    /// Get a reference to the underlying Vec.
    #[inline]
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Get a mutable reference to the underlying Vec (copy-on-write).
    ///
    /// If this buffer is shared with other buffers (e.g., through reshape),
    /// this will clone the data before returning a mutable reference.
    #[inline]
    pub fn data_mut(&mut self) -> &mut Vec<T> {
        Rc::make_mut(&mut self.data)
    }

    /// Consume the buffer and return the underlying Vec.
    ///
    /// If this is the last reference, no copy is made.
    /// Otherwise, the data is cloned.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        Rc::try_unwrap(self.data).unwrap_or_else(|rc| (*rc).clone())
    }

    /// Check if this buffer is shared with other buffers.
    ///
    /// Returns `true` if more than one buffer shares this data.
    #[inline]
    pub fn is_shared(&self) -> bool {
        Rc::strong_count(&self.data) > 1
    }

    /// Check if two buffers share the same underlying storage.
    #[inline]
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}

impl<T: Scalar> DataBuffer<T> for CpuBuffer<T> {
    #[inline]
    fn zeros(len: usize) -> Self {
        Self {
            data: Rc::new(vec![T::zero(); len]),
        }
    }

    #[inline]
    fn from_vec(data: Vec<T>) -> Self {
        Self {
            data: Rc::new(data),
        }
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
        Rc::make_mut(&mut self.data).as_mut_slice()
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
        &mut Rc::make_mut(&mut self.data)[i]
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

    #[test]
    fn test_cpu_buffer_shared_storage() {
        let buf1 = CpuBuffer::from_vec(vec![1.0, 2.0, 3.0]);
        let buf2 = buf1.clone();

        // Both buffers share the same underlying storage
        assert!(buf1.is_shared());
        assert!(buf2.is_shared());
        assert!(buf1.shares_storage_with(&buf2));

        // Data is equal
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn test_cpu_buffer_copy_on_write() {
        let buf1 = CpuBuffer::from_vec(vec![1.0, 2.0, 3.0]);
        let mut buf2 = buf1.clone();

        // Initially shared
        assert!(buf1.shares_storage_with(&buf2));

        // Mutation triggers copy
        buf2[1] = 5.0;

        // No longer shared after mutation
        assert!(!buf1.shares_storage_with(&buf2));
        assert!(!buf1.is_shared());
        assert!(!buf2.is_shared());

        // Data is different
        assert_eq!(buf1[1], 2.0);
        assert_eq!(buf2[1], 5.0);
    }

    #[test]
    fn test_cpu_buffer_not_shared_single_ref() {
        let buf = CpuBuffer::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(!buf.is_shared());
    }
}
