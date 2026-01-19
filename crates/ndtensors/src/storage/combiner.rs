//! Combiner storage for tensor index combining operations.
//!
//! This module provides the `Combiner` storage type which represents
//! index combining and uncombining operations in tensor networks.
//! Combiner tensors don't store actual data but define how multiple
//! indices should be combined into one (or vice versa).
//!
//! Mirrors NDTensors.jl's `Combiner <: TensorStorage{Number}` type.
//!
//! # Overview
//!
//! A combiner tensor has:
//! - One "combined" index (by convention the first index)
//! - Multiple "uncombined" indices
//!
//! Contracting a tensor with a combiner:
//! - **Combining**: Contract uncombined indices → combined index replaces them
//! - **Uncombining**: Contract combined index → uncombined indices replace it
//!
//! # Example
//!
//! ```
//! use ndtensors::storage::Combiner;
//!
//! // Create a combiner for combining 2 indices
//! let combiner = Combiner::new(vec![1], vec![1]);
//!
//! assert_eq!(combiner.blockperm(), &[1]);
//! assert_eq!(combiner.blockcomb(), &[1]);
//! assert!(!combiner.is_conj());
//! ```

use std::fmt;

/// Combiner storage - represents index combining/uncombining operations.
///
/// A combiner defines how to combine multiple tensor indices into a single
/// index (or the reverse uncombining operation). It stores the permutation
/// and combination patterns for blocks.
///
/// # Type Information
///
/// Unlike other storage types, Combiner doesn't store actual data. It stores
/// metadata about how to perform the combining operation:
///
/// * `perm` - Block permutation pattern
/// * `comb` - Block combination pattern
/// * `cind` - Combined index positions (default: [1])
/// * `is_conj` - Whether the combiner is conjugated
///
/// # Combining vs Uncombining
///
/// When contracting a tensor T with a combiner C:
/// - If the combined index of C is NOT in T's contracted indices → **combining**
/// - If the combined index of C IS in T's contracted indices → **uncombining**
#[derive(Debug, Clone, PartialEq)]
pub struct Combiner {
    /// Block permutation pattern.
    perm: Vec<usize>,
    /// Block combination pattern.
    comb: Vec<usize>,
    /// Combined index positions (by convention, position 0/1 is combined).
    cind: Vec<usize>,
    /// Whether the combiner is conjugated.
    is_conj: bool,
}

impl Default for Combiner {
    fn default() -> Self {
        Self::empty()
    }
}

impl Combiner {
    /// Create a new combiner with given permutation and combination patterns.
    ///
    /// # Arguments
    ///
    /// * `perm` - Block permutation pattern
    /// * `comb` - Block combination pattern
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::Combiner;
    ///
    /// let combiner = Combiner::new(vec![1], vec![1]);
    /// assert_eq!(combiner.blockperm(), &[1]);
    /// ```
    pub fn new(perm: Vec<usize>, comb: Vec<usize>) -> Self {
        Self {
            perm,
            comb,
            cind: vec![1], // By convention, combined index is at position 1 (0-indexed: 0)
            is_conj: false,
        }
    }

    /// Create a new combiner with full specification.
    ///
    /// # Arguments
    ///
    /// * `perm` - Block permutation pattern
    /// * `comb` - Block combination pattern
    /// * `cind` - Combined index positions
    /// * `is_conj` - Whether conjugated
    pub fn new_full(perm: Vec<usize>, comb: Vec<usize>, cind: Vec<usize>, is_conj: bool) -> Self {
        Self {
            perm,
            comb,
            cind,
            is_conj,
        }
    }

    /// Create an empty combiner.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::Combiner;
    ///
    /// let combiner = Combiner::empty();
    /// assert!(combiner.blockperm().is_empty());
    /// ```
    pub fn empty() -> Self {
        Self {
            perm: Vec::new(),
            comb: Vec::new(),
            cind: vec![1],
            is_conj: false,
        }
    }

    /// Get the block permutation pattern.
    #[inline]
    pub fn blockperm(&self) -> &[usize] {
        &self.perm
    }

    /// Get the block combination pattern.
    #[inline]
    pub fn blockcomb(&self) -> &[usize] {
        &self.comb
    }

    /// Get the combined index positions.
    #[inline]
    pub fn cinds(&self) -> &[usize] {
        &self.cind
    }

    /// Check if the combiner is conjugated.
    #[inline]
    pub fn is_conj(&self) -> bool {
        self.is_conj
    }

    /// Create a conjugated version of this combiner.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::Combiner;
    ///
    /// let combiner = Combiner::new(vec![1], vec![1]);
    /// assert!(!combiner.is_conj());
    ///
    /// let conj = combiner.conj();
    /// assert!(conj.is_conj());
    ///
    /// // Double conjugation returns to original
    /// let conj_conj = conj.conj();
    /// assert!(!conj_conj.is_conj());
    /// ```
    pub fn conj(&self) -> Self {
        Self {
            perm: self.perm.clone(),
            comb: self.comb.clone(),
            cind: self.cind.clone(),
            is_conj: !self.is_conj,
        }
    }

    /// Set the conjugation flag.
    pub fn set_conj(&mut self, is_conj: bool) {
        self.is_conj = is_conj;
    }

    /// Get the position of the combined index (0-indexed).
    ///
    /// By convention, the combined index is at position 0.
    #[inline]
    pub const fn combined_ind_position(&self) -> usize {
        0
    }

    /// Check if the storage is empty.
    ///
    /// Combiner storage is always considered "empty" in terms of data
    /// (it stores no actual tensor elements).
    #[inline]
    pub const fn is_empty(&self) -> bool {
        true
    }

    /// Get the length (number of stored elements).
    ///
    /// Combiner stores no data, so this always returns 0.
    #[inline]
    pub const fn len(&self) -> usize {
        0
    }

    /// Get the number of non-zero blocks.
    ///
    /// Combiner stores no data, so this always returns 0.
    #[inline]
    pub const fn nnzblocks(&self) -> usize {
        0
    }

    /// Get the number of non-zero elements.
    ///
    /// Combiner stores no data, so this always returns 0.
    #[inline]
    pub const fn nnz(&self) -> usize {
        0
    }
}

impl fmt::Display for Combiner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Combiner:")?;
        writeln!(f, "  Permutation of blocks: {:?}", self.perm)?;
        write!(f, "  Combination of blocks: {:?}", self.comb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combiner_new() {
        let combiner = Combiner::new(vec![1], vec![1]);
        assert_eq!(combiner.blockperm(), &[1]);
        assert_eq!(combiner.blockcomb(), &[1]);
        assert_eq!(combiner.cinds(), &[1]);
        assert!(!combiner.is_conj());
    }

    #[test]
    fn test_combiner_new_full() {
        let combiner = Combiner::new_full(vec![1, 2], vec![3, 4], vec![0], true);
        assert_eq!(combiner.blockperm(), &[1, 2]);
        assert_eq!(combiner.blockcomb(), &[3, 4]);
        assert_eq!(combiner.cinds(), &[0]);
        assert!(combiner.is_conj());
    }

    #[test]
    fn test_combiner_empty() {
        let combiner = Combiner::empty();
        assert!(combiner.blockperm().is_empty());
        assert!(combiner.blockcomb().is_empty());
        assert_eq!(combiner.cinds(), &[1]);
        assert!(!combiner.is_conj());
    }

    #[test]
    fn test_combiner_default() {
        let combiner: Combiner = Combiner::default();
        assert!(combiner.blockperm().is_empty());
    }

    #[test]
    fn test_combiner_conj() {
        let combiner = Combiner::new(vec![1], vec![1]);
        assert!(!combiner.is_conj());

        let conj = combiner.conj();
        assert!(conj.is_conj());
        assert_eq!(conj.blockperm(), combiner.blockperm());
        assert_eq!(conj.blockcomb(), combiner.blockcomb());

        let conj_conj = conj.conj();
        assert!(!conj_conj.is_conj());
    }

    #[test]
    fn test_combiner_set_conj() {
        let mut combiner = Combiner::new(vec![1], vec![1]);
        assert!(!combiner.is_conj());

        combiner.set_conj(true);
        assert!(combiner.is_conj());

        combiner.set_conj(false);
        assert!(!combiner.is_conj());
    }

    #[test]
    fn test_combiner_combined_ind_position() {
        let combiner = Combiner::new(vec![1], vec![1]);
        assert_eq!(combiner.combined_ind_position(), 0);
    }

    #[test]
    fn test_combiner_is_empty() {
        let combiner = Combiner::new(vec![1], vec![1]);
        assert!(combiner.is_empty());
    }

    #[test]
    fn test_combiner_len() {
        let combiner = Combiner::new(vec![1], vec![1]);
        assert_eq!(combiner.len(), 0);
    }

    #[test]
    fn test_combiner_nnzblocks() {
        let combiner = Combiner::new(vec![1], vec![1]);
        assert_eq!(combiner.nnzblocks(), 0);
    }

    #[test]
    fn test_combiner_nnz() {
        let combiner = Combiner::new(vec![1], vec![1]);
        assert_eq!(combiner.nnz(), 0);
    }

    #[test]
    fn test_combiner_equality() {
        let c1 = Combiner::new(vec![1, 2], vec![3, 4]);
        let c2 = Combiner::new(vec![1, 2], vec![3, 4]);
        let c3 = Combiner::new(vec![1], vec![3, 4]);

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_combiner_clone() {
        let c1 = Combiner::new(vec![1, 2], vec![3, 4]);
        let c2 = c1.clone();
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_combiner_display() {
        let combiner = Combiner::new(vec![1, 2], vec![3, 4]);
        let display = format!("{}", combiner);
        assert!(display.contains("Combiner"));
        assert!(display.contains("Permutation of blocks"));
        assert!(display.contains("Combination of blocks"));
    }

    #[test]
    fn test_combiner_debug() {
        let combiner = Combiner::new(vec![1], vec![1]);
        let debug = format!("{:?}", combiner);
        assert!(debug.contains("Combiner"));
    }
}
