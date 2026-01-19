//! Diagonal storage for tensor data.
//!
//! This module provides the `Diag` storage type which holds only diagonal
//! elements of a tensor. For a tensor of shape [d0, d1, ..., dn], the diagonal
//! storage holds min(d0, d1, ..., dn) elements.
//!
//! Mirrors NDTensors.jl's `Diag{ElT, VecT}` storage type.

use std::marker::PhantomData;

use crate::scalar::Scalar;
use crate::storage::buffer::{CpuBuffer, DataBuffer};

/// Diagonal storage - stores only diagonal elements.
///
/// For a tensor with shape [d0, d1, ..., dn], stores min(d0, d1, ..., dn) elements
/// representing the diagonal entries where all indices are equal.
///
/// This mirrors NDTensors.jl's `Diag{ElT, VecT}` structure.
///
/// # Type Parameters
///
/// * `ElT` - Element type (e.g., f64, c64)
/// * `D` - Data buffer type, defaults to `CpuBuffer<ElT>`
///
/// # Variants
///
/// NDTensors.jl supports two variants:
/// - Non-uniform: `VecT <: AbstractVector{ElT}` - different values per diagonal element
/// - Uniform: `VecT <: Number` - single scalar value for all diagonal elements
///
/// Currently, only the non-uniform variant is implemented.
///
/// # Example
///
/// ```
/// use ndtensors::storage::Diag;
///
/// // Create diagonal storage with 3 elements
/// let diag: Diag<f64> = Diag::from_vec(vec![1.0, 2.0, 3.0]);
/// assert_eq!(diag.len(), 3);
/// assert_eq!(diag[0], 1.0);
/// assert_eq!(diag[2], 3.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Diag<ElT: Scalar, D: DataBuffer<ElT> = CpuBuffer<ElT>> {
    data: D,
    _phantom: PhantomData<ElT>,
}

/// Type alias for CPU-backed diagonal storage.
pub type CpuDiag<ElT> = Diag<ElT, CpuBuffer<ElT>>;

impl<ElT: Scalar, D: DataBuffer<ElT>> Diag<ElT, D> {
    /// Create diagonal storage with given length, zero-initialized.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::Diag;
    ///
    /// let diag: Diag<f64> = Diag::zeros(5);
    /// assert_eq!(diag.len(), 5);
    /// for i in 0..5 {
    ///     assert_eq!(diag[i], 0.0);
    /// }
    /// ```
    pub fn zeros(len: usize) -> Self {
        Self {
            data: D::zeros(len),
            _phantom: PhantomData,
        }
    }

    /// Create diagonal storage from existing vector (takes ownership).
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::Diag;
    ///
    /// let diag: Diag<f64> = Diag::from_vec(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(diag.len(), 3);
    /// assert_eq!(diag[1], 2.0);
    /// ```
    pub fn from_vec(data: Vec<ElT>) -> Self {
        Self {
            data: D::from_vec(data),
            _phantom: PhantomData,
        }
    }

    /// Create diagonal storage from an existing data buffer.
    pub fn from_buffer(data: D) -> Self {
        Self {
            data,
            _phantom: PhantomData,
        }
    }

    /// Create diagonal storage with all elements set to a single value.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::Diag;
    ///
    /// let diag: Diag<f64> = Diag::fill(3, 5.0);
    /// assert_eq!(diag.len(), 3);
    /// assert_eq!(diag[0], 5.0);
    /// assert_eq!(diag[2], 5.0);
    /// ```
    pub fn fill(len: usize, value: ElT) -> Self {
        Self {
            data: D::from_vec(vec![value; len]),
            _phantom: PhantomData,
        }
    }

    /// Create diagonal storage representing an identity matrix diagonal.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::Diag;
    ///
    /// let diag: Diag<f64> = Diag::identity(4);
    /// assert_eq!(diag.len(), 4);
    /// for i in 0..4 {
    ///     assert_eq!(diag[i], 1.0);
    /// }
    /// ```
    pub fn identity(len: usize) -> Self {
        Self::fill(len, ElT::one())
    }

    /// Length of storage (number of diagonal elements).
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
    #[inline]
    pub fn view(&self) -> Self {
        Self {
            data: self.data.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<ElT: Scalar, D: DataBuffer<ElT>> std::ops::Index<usize> for Diag<ElT, D> {
    type Output = ElT;

    #[inline]
    fn index(&self, i: usize) -> &ElT {
        &self.data.as_slice()[i]
    }
}

impl<ElT: Scalar, D: DataBuffer<ElT>> std::ops::IndexMut<usize> for Diag<ElT, D> {
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
        let d: Diag<f64> = Diag::zeros(5);
        assert_eq!(d.len(), 5);
        assert!(!d.is_empty());
        for i in 0..5 {
            assert_eq!(d[i], 0.0);
        }
    }

    #[test]
    fn test_from_vec() {
        let d: Diag<f64> = Diag::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(d.len(), 3);
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 2.0);
        assert_eq!(d[2], 3.0);
    }

    #[test]
    fn test_fill() {
        let d: Diag<f64> = Diag::fill(4, 7.0);
        assert_eq!(d.len(), 4);
        for i in 0..4 {
            assert_eq!(d[i], 7.0);
        }
    }

    #[test]
    fn test_identity() {
        let d: Diag<f64> = Diag::identity(3);
        assert_eq!(d.len(), 3);
        for i in 0..3 {
            assert_eq!(d[i], 1.0);
        }
    }

    #[test]
    fn test_index_mut() {
        let mut d: Diag<f64> = Diag::zeros(3);
        d[1] = 5.0;
        assert_eq!(d[1], 5.0);
    }

    #[test]
    fn test_as_slice() {
        let d: Diag<f64> = Diag::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(d.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_buffer() {
        let buf = CpuBuffer::from_vec(vec![1.0, 2.0, 3.0]);
        let d: Diag<f64> = Diag::from_buffer(buf);
        assert_eq!(d.len(), 3);
        assert_eq!(d[0], 1.0);
    }

    #[test]
    fn test_cpu_diag_alias() {
        let d: CpuDiag<f64> = CpuDiag::zeros(5);
        assert_eq!(d.len(), 5);
    }

    #[test]
    fn test_complex_diag() {
        use crate::scalar::c64;
        let d: Diag<c64> = Diag::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)]);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0], c64::new(1.0, 2.0));
        assert_eq!(d[1], c64::new(3.0, 4.0));
    }
}
