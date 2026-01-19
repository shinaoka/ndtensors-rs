//! CombinerTensor - Tensor type for index combining operations.
//!
//! This module provides `CombinerTensor`, a tensor wrapper for the Combiner
//! storage type. Combiner tensors represent index combining and uncombining
//! operations in tensor networks.
//!
//! # Overview
//!
//! A combiner tensor has:
//! - Shape: (combined_dim, uncombined_dim_1, uncombined_dim_2, ...)
//! - By convention, the combined index is at position 0
//! - No actual data storage (uses Combiner storage)
//!
//! # Design
//!
//! CombinerTensor mirrors NDTensors.jl's `CombinerTensor` type alias:
//! ```julia
//! const CombinerTensor{ElT, N, StoreT, IndsT} =
//!     Tensor{ElT, N, StoreT, IndsT} where {StoreT <: Combiner}
//! ```
//!
//! # Example
//!
//! ```
//! use ndtensors::combiner_tensor::CombinerTensor;
//!
//! // Create a combiner for combining two 2x2 indices into a 4-dimensional index
//! let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
//!
//! assert_eq!(tensor.shape(), &[4, 2, 2]);
//! assert_eq!(tensor.ndim(), 3);
//! assert_eq!(tensor.combined_ind_position(), 0);
//! assert_eq!(tensor.combined_ind_dim(), 4);
//! ```
//!
//! # Combining and Uncombining
//!
//! When contracting with a combiner:
//! - **Combining**: Multiple indices → single combined index
//! - **Uncombining**: Single combined index → multiple indices
//!
//! Note: Full combining/uncombining contraction operations are planned for future implementation.

use crate::storage::Combiner;

/// A combiner tensor for index combining operations.
///
/// CombinerTensor wraps a Combiner storage with shape information.
/// The shape represents the dimensions of the combined and uncombined indices.
///
/// # Convention
///
/// - The first index (position 0) is the combined index
/// - Remaining indices are the uncombined indices
#[derive(Clone, Debug, PartialEq)]
pub struct CombinerTensor {
    storage: Combiner,
    shape: Vec<usize>,
}

impl CombinerTensor {
    /// Create a new combiner tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the combiner tensor: [combined_dim, uncombined_dim_1, ...]
    /// * `blockperm` - Block permutation pattern
    /// * `blockcomb` - Block combination pattern
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::combiner_tensor::CombinerTensor;
    ///
    /// // Combine two 2x2 indices into one 4-dimensional index
    /// let combiner = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
    /// assert_eq!(combiner.shape(), &[4, 2, 2]);
    /// ```
    pub fn new(shape: &[usize], blockperm: Vec<usize>, blockcomb: Vec<usize>) -> Self {
        Self {
            storage: Combiner::new(blockperm, blockcomb),
            shape: shape.to_vec(),
        }
    }

    /// Create a combiner tensor from existing storage.
    pub fn from_storage(storage: Combiner, shape: Vec<usize>) -> Self {
        Self { storage, shape }
    }

    /// Create a simple combiner for combining indices.
    ///
    /// # Arguments
    ///
    /// * `combined_dim` - Dimension of the combined index
    /// * `uncombined_dims` - Dimensions of the uncombined indices
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::combiner_tensor::CombinerTensor;
    ///
    /// // Combine three dimensions (2, 2, 3) into one (12)
    /// let combiner = CombinerTensor::simple(12, &[2, 2, 3]);
    /// assert_eq!(combiner.shape(), &[12, 2, 2, 3]);
    /// assert_eq!(combiner.combined_ind_dim(), 12);
    /// ```
    pub fn simple(combined_dim: usize, uncombined_dims: &[usize]) -> Self {
        let mut shape = vec![combined_dim];
        shape.extend_from_slice(uncombined_dims);
        Self {
            storage: Combiner::new(vec![1], vec![1]),
            shape,
        }
    }

    /// Get the shape of the combiner tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the underlying combiner storage.
    #[inline]
    pub fn storage(&self) -> &Combiner {
        &self.storage
    }

    /// Get mutable access to the underlying storage.
    #[inline]
    pub fn storage_mut(&mut self) -> &mut Combiner {
        &mut self.storage
    }

    /// Get the position of the combined index (always 0 by convention).
    #[inline]
    pub const fn combined_ind_position(&self) -> usize {
        0
    }

    /// Get the dimension of the combined index.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::combiner_tensor::CombinerTensor;
    ///
    /// let combiner = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
    /// assert_eq!(combiner.combined_ind_dim(), 4);
    /// ```
    pub fn combined_ind_dim(&self) -> usize {
        self.shape[self.combined_ind_position()]
    }

    /// Get the dimensions of the uncombined indices.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::combiner_tensor::CombinerTensor;
    ///
    /// let combiner = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
    /// assert_eq!(combiner.uncombined_ind_dims(), &[2, 2]);
    /// ```
    pub fn uncombined_ind_dims(&self) -> &[usize] {
        &self.shape[1..]
    }

    /// Get the number of uncombined indices.
    pub fn num_uncombined(&self) -> usize {
        self.ndim() - 1
    }

    /// Check if the combiner is conjugated.
    #[inline]
    pub fn is_conj(&self) -> bool {
        self.storage.is_conj()
    }

    /// Create a conjugated version of this combiner tensor.
    pub fn conj(&self) -> Self {
        Self {
            storage: self.storage.conj(),
            shape: self.shape.clone(),
        }
    }

    /// Validate that the combined dimension equals the product of uncombined dimensions.
    ///
    /// # Returns
    ///
    /// `true` if `combined_dim == product(uncombined_dims)`
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::combiner_tensor::CombinerTensor;
    ///
    /// let valid = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
    /// assert!(valid.is_valid_shape());
    ///
    /// let invalid = CombinerTensor::new(&[5, 2, 2], vec![1], vec![1]);
    /// assert!(!invalid.is_valid_shape());
    /// ```
    pub fn is_valid_shape(&self) -> bool {
        let combined = self.combined_ind_dim();
        let product: usize = self.uncombined_ind_dims().iter().product();
        combined == product
    }
}

impl std::fmt::Display for CombinerTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CombinerTensor(shape={:?}, combined_dim={}, uncombined_dims={:?})",
            self.shape,
            self.combined_ind_dim(),
            self.uncombined_ind_dims()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combiner_tensor_new() {
        let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        assert_eq!(tensor.shape(), &[4, 2, 2]);
        assert_eq!(tensor.ndim(), 3);
    }

    #[test]
    fn test_combiner_tensor_simple() {
        let tensor = CombinerTensor::simple(12, &[2, 2, 3]);
        assert_eq!(tensor.shape(), &[12, 2, 2, 3]);
        assert_eq!(tensor.combined_ind_dim(), 12);
        assert_eq!(tensor.uncombined_ind_dims(), &[2, 2, 3]);
    }

    #[test]
    fn test_combiner_tensor_combined_ind_position() {
        let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        assert_eq!(tensor.combined_ind_position(), 0);
    }

    #[test]
    fn test_combiner_tensor_combined_ind_dim() {
        let tensor = CombinerTensor::new(&[8, 2, 4], vec![1], vec![1]);
        assert_eq!(tensor.combined_ind_dim(), 8);
    }

    #[test]
    fn test_combiner_tensor_uncombined_ind_dims() {
        let tensor = CombinerTensor::new(&[6, 2, 3], vec![1], vec![1]);
        assert_eq!(tensor.uncombined_ind_dims(), &[2, 3]);
    }

    #[test]
    fn test_combiner_tensor_num_uncombined() {
        let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        assert_eq!(tensor.num_uncombined(), 2);

        let tensor2 = CombinerTensor::new(&[8, 2, 2, 2], vec![1], vec![1]);
        assert_eq!(tensor2.num_uncombined(), 3);
    }

    #[test]
    fn test_combiner_tensor_is_conj() {
        let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        assert!(!tensor.is_conj());

        let conj = tensor.conj();
        assert!(conj.is_conj());
    }

    #[test]
    fn test_combiner_tensor_conj() {
        let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        let conj = tensor.conj();

        assert!(conj.is_conj());
        assert_eq!(conj.shape(), tensor.shape());

        let conj_conj = conj.conj();
        assert!(!conj_conj.is_conj());
    }

    #[test]
    fn test_combiner_tensor_is_valid_shape() {
        // Valid: 4 == 2 * 2
        let valid = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        assert!(valid.is_valid_shape());

        // Valid: 12 == 2 * 2 * 3
        let valid2 = CombinerTensor::simple(12, &[2, 2, 3]);
        assert!(valid2.is_valid_shape());

        // Invalid: 5 != 2 * 2
        let invalid = CombinerTensor::new(&[5, 2, 2], vec![1], vec![1]);
        assert!(!invalid.is_valid_shape());
    }

    #[test]
    fn test_combiner_tensor_from_storage() {
        let storage = Combiner::new(vec![1, 2], vec![3, 4]);
        let tensor = CombinerTensor::from_storage(storage.clone(), vec![6, 2, 3]);

        assert_eq!(tensor.shape(), &[6, 2, 3]);
        assert_eq!(tensor.storage().blockperm(), storage.blockperm());
    }

    #[test]
    fn test_combiner_tensor_storage_mut() {
        let mut tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        assert!(!tensor.is_conj());

        tensor.storage_mut().set_conj(true);
        assert!(tensor.is_conj());
    }

    #[test]
    fn test_combiner_tensor_equality() {
        let t1 = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        let t2 = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        let t3 = CombinerTensor::new(&[6, 2, 3], vec![1], vec![1]);

        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    #[test]
    fn test_combiner_tensor_clone() {
        let t1 = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        let t2 = t1.clone();
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_combiner_tensor_display() {
        let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        let display = format!("{}", tensor);

        assert!(display.contains("CombinerTensor"));
        assert!(display.contains("shape="));
        assert!(display.contains("combined_dim="));
        assert!(display.contains("uncombined_dims="));
    }

    #[test]
    fn test_combiner_tensor_debug() {
        let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
        let debug = format!("{:?}", tensor);
        assert!(debug.contains("CombinerTensor"));
    }
}
