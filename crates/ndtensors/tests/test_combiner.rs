//! Tests for Combiner storage and CombinerTensor.
//!
//! These tests cover the basic Combiner functionality, mirroring concepts from
//! NDTensors.jl's test_combiner.jl. Full combining/uncombining contraction
//! operations are planned for future implementation.
//!
//! # Coverage
//!
//! - Combiner storage creation and properties
//! - CombinerTensor creation and shape handling
//! - Conjugation operations
//! - Shape validation
//!
//! # Note
//!
//! The Julia tests in test_combiner.jl test actual tensor contractions with
//! combiners (combining and uncombining operations). These are not yet
//! implemented in Rust. The tests here cover the foundational types.

use ndtensors::combiner_tensor::CombinerTensor;
use ndtensors::storage::Combiner;

// ============================================================================
// Combiner Storage Tests
// ============================================================================

/// Test Combiner creation with permutation and combination patterns.
#[test]
fn test_combiner_new() {
    let combiner = Combiner::new(vec![1], vec![1]);

    assert_eq!(combiner.blockperm(), &[1]);
    assert_eq!(combiner.blockcomb(), &[1]);
    assert!(!combiner.is_conj());
}

/// Test Combiner with full specification.
#[test]
fn test_combiner_new_full() {
    let combiner = Combiner::new_full(vec![1, 2, 3], vec![4, 5], vec![0], true);

    assert_eq!(combiner.blockperm(), &[1, 2, 3]);
    assert_eq!(combiner.blockcomb(), &[4, 5]);
    assert_eq!(combiner.cinds(), &[0]);
    assert!(combiner.is_conj());
}

/// Test empty Combiner.
#[test]
fn test_combiner_empty() {
    let combiner = Combiner::empty();

    assert!(combiner.blockperm().is_empty());
    assert!(combiner.blockcomb().is_empty());
    assert!(!combiner.is_conj());
}

/// Test Combiner default is empty.
#[test]
fn test_combiner_default() {
    let combiner: Combiner = Default::default();

    assert!(combiner.blockperm().is_empty());
    assert!(combiner.blockcomb().is_empty());
}

/// Test Combiner conjugation.
#[test]
fn test_combiner_conj() {
    let combiner = Combiner::new(vec![1, 2], vec![3, 4]);
    assert!(!combiner.is_conj());

    let conj = combiner.conj();
    assert!(conj.is_conj());
    assert_eq!(conj.blockperm(), combiner.blockperm());
    assert_eq!(conj.blockcomb(), combiner.blockcomb());

    // Double conjugation
    let conj_conj = conj.conj();
    assert!(!conj_conj.is_conj());
}

/// Test Combiner set_conj.
#[test]
fn test_combiner_set_conj() {
    let mut combiner = Combiner::new(vec![1], vec![1]);

    combiner.set_conj(true);
    assert!(combiner.is_conj());

    combiner.set_conj(false);
    assert!(!combiner.is_conj());
}

/// Test Combiner combined_ind_position.
#[test]
fn test_combiner_combined_ind_position() {
    let combiner = Combiner::new(vec![1], vec![1]);
    assert_eq!(combiner.combined_ind_position(), 0);
}

/// Test Combiner is_empty always returns true.
#[test]
fn test_combiner_is_empty() {
    let combiner = Combiner::new(vec![1, 2, 3], vec![4, 5, 6]);
    assert!(combiner.is_empty());
}

/// Test Combiner len always returns 0.
#[test]
fn test_combiner_len() {
    let combiner = Combiner::new(vec![1, 2], vec![3, 4]);
    assert_eq!(combiner.len(), 0);
}

/// Test Combiner nnzblocks always returns 0.
#[test]
fn test_combiner_nnzblocks() {
    let combiner = Combiner::new(vec![1], vec![1]);
    assert_eq!(combiner.nnzblocks(), 0);
}

/// Test Combiner nnz always returns 0.
#[test]
fn test_combiner_nnz() {
    let combiner = Combiner::new(vec![1], vec![1]);
    assert_eq!(combiner.nnz(), 0);
}

/// Test Combiner equality.
#[test]
fn test_combiner_equality() {
    let c1 = Combiner::new(vec![1, 2], vec![3, 4]);
    let c2 = Combiner::new(vec![1, 2], vec![3, 4]);
    let c3 = Combiner::new(vec![1], vec![3, 4]);
    let c4 = Combiner::new(vec![1, 2], vec![5, 6]);

    assert_eq!(c1, c2);
    assert_ne!(c1, c3);
    assert_ne!(c1, c4);
}

/// Test Combiner clone.
#[test]
fn test_combiner_clone() {
    let c1 = Combiner::new(vec![1, 2], vec![3, 4]);
    let c2 = c1.clone();
    assert_eq!(c1, c2);
}

/// Test Combiner display.
#[test]
fn test_combiner_display() {
    let combiner = Combiner::new(vec![1, 2], vec![3, 4]);
    let display = format!("{}", combiner);

    assert!(display.contains("Combiner"));
    assert!(display.contains("Permutation of blocks"));
    assert!(display.contains("Combination of blocks"));
}

/// Test Combiner debug.
#[test]
fn test_combiner_debug() {
    let combiner = Combiner::new(vec![1], vec![1]);
    let debug = format!("{:?}", combiner);
    assert!(debug.contains("Combiner"));
}

// ============================================================================
// CombinerTensor Tests
// ============================================================================

/// Test CombinerTensor creation.
/// Mirrors: tensor(Combiner([1], [1]), combiner_tensor_inds)
#[test]
fn test_combiner_tensor_new() {
    // Shape: (combined=4, uncombined1=2, uncombined2=2)
    let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);

    assert_eq!(tensor.shape(), &[4, 2, 2]);
    assert_eq!(tensor.ndim(), 3);
    assert_eq!(tensor.combined_ind_position(), 0);
    assert_eq!(tensor.combined_ind_dim(), 4);
    assert_eq!(tensor.uncombined_ind_dims(), &[2, 2]);
}

/// Test CombinerTensor::simple constructor.
#[test]
fn test_combiner_tensor_simple() {
    // Combine three 2-dimensional indices into one 8-dimensional index
    let tensor = CombinerTensor::simple(8, &[2, 2, 2]);

    assert_eq!(tensor.shape(), &[8, 2, 2, 2]);
    assert_eq!(tensor.combined_ind_dim(), 8);
    assert_eq!(tensor.uncombined_ind_dims(), &[2, 2, 2]);
    assert!(tensor.is_valid_shape());
}

/// Test CombinerTensor for combining two indices.
/// Mirrors Julia: combiner_tensor_inds = (d^2, d, d) where d=2
#[test]
fn test_combiner_tensor_two_indices() {
    // d = 2, so combined = d^2 = 4, uncombined = (d, d) = (2, 2)
    let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);

    assert_eq!(tensor.combined_ind_dim(), 4);
    assert_eq!(tensor.uncombined_ind_dims(), &[2, 2]);
    assert_eq!(tensor.num_uncombined(), 2);
    assert!(tensor.is_valid_shape());
}

/// Test CombinerTensor shape validation with valid shape.
#[test]
fn test_combiner_tensor_valid_shape() {
    // 6 = 2 * 3
    let valid = CombinerTensor::new(&[6, 2, 3], vec![1], vec![1]);
    assert!(valid.is_valid_shape());

    // 24 = 2 * 3 * 4
    let valid2 = CombinerTensor::new(&[24, 2, 3, 4], vec![1], vec![1]);
    assert!(valid2.is_valid_shape());
}

/// Test CombinerTensor shape validation with invalid shape.
#[test]
fn test_combiner_tensor_invalid_shape() {
    // 5 != 2 * 2
    let invalid = CombinerTensor::new(&[5, 2, 2], vec![1], vec![1]);
    assert!(!invalid.is_valid_shape());

    // 10 != 2 * 3
    let invalid2 = CombinerTensor::new(&[10, 2, 3], vec![1], vec![1]);
    assert!(!invalid2.is_valid_shape());
}

/// Test CombinerTensor conjugation.
#[test]
fn test_combiner_tensor_conj() {
    let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
    assert!(!tensor.is_conj());

    let conj = tensor.conj();
    assert!(conj.is_conj());
    assert_eq!(conj.shape(), tensor.shape());

    let conj_conj = conj.conj();
    assert!(!conj_conj.is_conj());
}

/// Test CombinerTensor from_storage.
#[test]
fn test_combiner_tensor_from_storage() {
    let storage = Combiner::new(vec![1, 2], vec![3, 4]);
    let tensor = CombinerTensor::from_storage(storage.clone(), vec![6, 2, 3]);

    assert_eq!(tensor.shape(), &[6, 2, 3]);
    assert_eq!(tensor.storage().blockperm(), &[1, 2]);
}

/// Test CombinerTensor equality.
#[test]
fn test_combiner_tensor_equality() {
    let t1 = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
    let t2 = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
    let t3 = CombinerTensor::new(&[6, 2, 3], vec![1], vec![1]);

    assert_eq!(t1, t2);
    assert_ne!(t1, t3);
}

/// Test CombinerTensor clone.
#[test]
fn test_combiner_tensor_clone() {
    let t1 = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
    let t2 = t1.clone();
    assert_eq!(t1, t2);
}

/// Test CombinerTensor display.
#[test]
fn test_combiner_tensor_display() {
    let tensor = CombinerTensor::new(&[4, 2, 2], vec![1], vec![1]);
    let display = format!("{}", tensor);

    assert!(display.contains("CombinerTensor"));
    assert!(display.contains("shape="));
    assert!(display.contains("combined_dim=4"));
    assert!(display.contains("uncombined_dims="));
}

/// Test CombinerTensor with single uncombined index.
#[test]
fn test_combiner_tensor_single_uncombined() {
    let tensor = CombinerTensor::new(&[3, 3], vec![1], vec![1]);

    assert_eq!(tensor.combined_ind_dim(), 3);
    assert_eq!(tensor.uncombined_ind_dims(), &[3]);
    assert_eq!(tensor.num_uncombined(), 1);
    assert!(tensor.is_valid_shape());
}

/// Test CombinerTensor with many uncombined indices.
#[test]
fn test_combiner_tensor_many_uncombined() {
    // 24 = 2 * 2 * 2 * 3
    let tensor = CombinerTensor::new(&[24, 2, 2, 2, 3], vec![1], vec![1]);

    assert_eq!(tensor.combined_ind_dim(), 24);
    assert_eq!(tensor.uncombined_ind_dims(), &[2, 2, 2, 3]);
    assert_eq!(tensor.num_uncombined(), 4);
    assert!(tensor.is_valid_shape());
}
