//! Dense storage for tensor data.

use crate::scalar::Scalar;

/// Dense storage - contiguous array of elements in column-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Dense<ElT: Scalar> {
    data: Vec<ElT>,
}

impl<ElT: Scalar> Dense<ElT> {
    /// Create dense storage with given length, zero-initialized.
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![ElT::zero(); len],
        }
    }

    /// Create dense storage from existing vector (takes ownership).
    pub fn from_vec(data: Vec<ElT>) -> Self {
        Self { data }
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
        &self.data
    }

    /// Get mutable slice of data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [ElT] {
        &mut self.data
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
}

impl<ElT: Scalar> std::ops::Index<usize> for Dense<ElT> {
    type Output = ElT;

    #[inline]
    fn index(&self, i: usize) -> &ElT {
        &self.data[i]
    }
}

impl<ElT: Scalar> std::ops::IndexMut<usize> for Dense<ElT> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut ElT {
        &mut self.data[i]
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
        let d = Dense::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(d.len(), 3);
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 2.0);
        assert_eq!(d[2], 3.0);
    }

    #[test]
    fn test_index_mut() {
        let mut d = Dense::zeros(3);
        d[1] = 5.0;
        assert_eq!(d[1], 5.0);
    }

    #[test]
    fn test_as_slice() {
        let d = Dense::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(d.as_slice(), &[1.0, 2.0, 3.0]);
    }
}
