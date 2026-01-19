//! Empty storage for tensor data.
//!
//! This module provides `EmptyStorage<ElT>`, a storage type that represents
//! an empty or uninitialized tensor. EmptyStorage has no actual data but
//! maintains a nominal element type.
//!
//! This mirrors NDTensors.jl's `EmptyStorage{ElT, StoreT}` type.
//!
//! # Design
//!
//! EmptyStorage is useful for:
//! - Representing tensors before allocation
//! - Structurally sparse tensors with no non-zero blocks
//! - Lazy evaluation patterns
//!
//! # Example
//!
//! ```
//! use ndtensors::storage::EmptyStorage;
//!
//! let empty: EmptyStorage<f64> = EmptyStorage::new();
//! assert_eq!(empty.len(), 0);
//! assert!(empty.is_empty());
//! ```

use std::fmt;
use std::marker::PhantomData;

use crate::empty_number::EmptyNumber;

/// Empty storage - holds no data but maintains element type information.
///
/// This represents an empty or uninitialized tensor storage. Indexing into
/// EmptyStorage always returns the zero element (or EmptyNumber for
/// EmptyNumber element type).
///
/// # Type Parameters
///
/// * `ElT` - Element type (e.g., f64, EmptyNumber)
#[derive(Clone)]
pub struct EmptyStorage<ElT> {
    _phantom: PhantomData<ElT>,
}

impl<ElT> fmt::Debug for EmptyStorage<ElT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EmptyStorage<{}>", std::any::type_name::<ElT>())
    }
}

impl<ElT> Default for EmptyStorage<ElT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<ElT> EmptyStorage<ElT> {
    /// Create a new empty storage.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::EmptyStorage;
    ///
    /// let empty: EmptyStorage<f64> = EmptyStorage::new();
    /// assert_eq!(empty.len(), 0);
    /// ```
    pub const fn new() -> Self {
        EmptyStorage {
            _phantom: PhantomData,
        }
    }

    /// Length of storage (always 0).
    #[inline]
    pub const fn len(&self) -> usize {
        0
    }

    /// Check if storage is empty (always true).
    #[inline]
    pub const fn is_empty(&self) -> bool {
        true
    }

    /// Get immutable slice of data (always empty).
    #[inline]
    pub fn as_slice(&self) -> &[ElT] {
        &[]
    }

    /// Get mutable slice of data (always empty).
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [ElT] {
        &mut []
    }

    /// Number of non-zero blocks (always 0).
    #[inline]
    pub const fn nnzblocks(&self) -> usize {
        0
    }

    /// Number of non-zero elements (always 0).
    #[inline]
    pub const fn nnz(&self) -> usize {
        0
    }

    /// Get the zero element (returns a new EmptyStorage).
    pub fn zero(&self) -> Self {
        EmptyStorage::new()
    }
}

impl<ElT: PartialEq> PartialEq for EmptyStorage<ElT> {
    fn eq(&self, _other: &Self) -> bool {
        true // All empty storages of the same type are equal
    }
}

impl<ElT: Eq> Eq for EmptyStorage<ElT> {}

impl<ElT> fmt::Display for EmptyStorage<ElT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EmptyStorage<{}>", std::any::type_name::<ElT>())
    }
}

// Special implementations for EmptyNumber element type

impl EmptyStorage<EmptyNumber> {
    /// Create an empty storage with EmptyNumber element type.
    pub const fn empty() -> Self {
        EmptyStorage::new()
    }
}

/// Type alias for EmptyStorage with EmptyNumber elements.
pub type EmptyNumberStorage = EmptyStorage<EmptyNumber>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_storage_new() {
        let empty: EmptyStorage<f64> = EmptyStorage::new();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_empty_storage_default() {
        let empty: EmptyStorage<f64> = EmptyStorage::default();
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_empty_storage_as_slice() {
        let empty: EmptyStorage<f64> = EmptyStorage::new();
        assert_eq!(empty.as_slice().len(), 0);
    }

    #[test]
    fn test_empty_storage_as_mut_slice() {
        let mut empty: EmptyStorage<f64> = EmptyStorage::new();
        assert_eq!(empty.as_mut_slice().len(), 0);
    }

    #[test]
    fn test_empty_storage_nnzblocks() {
        let empty: EmptyStorage<f64> = EmptyStorage::new();
        assert_eq!(empty.nnzblocks(), 0);
    }

    #[test]
    fn test_empty_storage_nnz() {
        let empty: EmptyStorage<f64> = EmptyStorage::new();
        assert_eq!(empty.nnz(), 0);
    }

    #[test]
    fn test_empty_storage_zero() {
        let empty: EmptyStorage<f64> = EmptyStorage::new();
        let zero = empty.zero();
        assert!(zero.is_empty());
    }

    #[test]
    fn test_empty_storage_equality() {
        let empty1: EmptyStorage<f64> = EmptyStorage::new();
        let empty2: EmptyStorage<f64> = EmptyStorage::new();
        assert_eq!(empty1, empty2);
    }

    #[test]
    fn test_empty_storage_display() {
        let empty: EmptyStorage<f64> = EmptyStorage::new();
        let display = format!("{}", empty);
        assert!(display.contains("EmptyStorage"));
    }

    #[test]
    fn test_empty_storage_debug() {
        let empty: EmptyStorage<f64> = EmptyStorage::new();
        let debug = format!("{:?}", empty);
        assert!(debug.contains("EmptyStorage"));
    }

    #[test]
    fn test_empty_number_storage() {
        let empty: EmptyStorage<EmptyNumber> = EmptyStorage::empty();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_empty_number_storage_alias() {
        let empty: EmptyNumberStorage = EmptyNumberStorage::new();
        assert!(empty.is_empty());
    }
}
