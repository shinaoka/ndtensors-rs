//! Tests for EmptyStorage.
//!
//! These tests mirror NDTensors.jl's test_emptystorage.jl, covering:
//! - EmptyStorage creation
//! - Basic properties (size, length, is_empty)
//! - Zero initialization
//! - Number of non-zero blocks
//! - Equality

use ndtensors::EmptyNumber;
use ndtensors::storage::EmptyStorage;

/// Test EmptyStorage creation with f64 element type.
/// Mirrors: T = Tensor(EmptyStorage(NDTensors.EmptyNumber), (2, 2))
#[test]
fn test_emptystorage_new_f64() {
    let empty: EmptyStorage<f64> = EmptyStorage::new();
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}

/// Test EmptyStorage creation with EmptyNumber element type.
#[test]
fn test_emptystorage_new_emptynumber() {
    let empty: EmptyStorage<EmptyNumber> = EmptyStorage::new();
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}

/// Test EmptyStorage default.
#[test]
fn test_emptystorage_default() {
    let empty: EmptyStorage<f64> = EmptyStorage::default();
    assert_eq!(empty.len(), 0);
}

/// Test EmptyStorage as_slice returns empty slice.
#[test]
fn test_emptystorage_as_slice() {
    let empty: EmptyStorage<f64> = EmptyStorage::new();
    let slice = empty.as_slice();
    assert_eq!(slice.len(), 0);
}

/// Test EmptyStorage as_mut_slice returns empty slice.
#[test]
fn test_emptystorage_as_mut_slice() {
    let mut empty: EmptyStorage<f64> = EmptyStorage::new();
    let slice = empty.as_mut_slice();
    assert_eq!(slice.len(), 0);
}

/// Test EmptyStorage nnzblocks is 0.
/// Mirrors: @test blockoffsets(T) == BlockOffsets{2}()
#[test]
fn test_emptystorage_nnzblocks() {
    let empty: EmptyStorage<f64> = EmptyStorage::new();
    assert_eq!(empty.nnzblocks(), 0);
}

/// Test EmptyStorage nnz is 0.
#[test]
fn test_emptystorage_nnz() {
    let empty: EmptyStorage<f64> = EmptyStorage::new();
    assert_eq!(empty.nnz(), 0);
}

/// Test EmptyStorage zero returns itself.
/// Mirrors: @test zero(T) isa typeof(T)
#[test]
fn test_emptystorage_zero() {
    let empty: EmptyStorage<f64> = EmptyStorage::new();
    let zero = empty.zero();
    assert!(zero.is_empty());
}

/// Test EmptyStorage equality.
#[test]
fn test_emptystorage_equality() {
    let empty1: EmptyStorage<f64> = EmptyStorage::new();
    let empty2: EmptyStorage<f64> = EmptyStorage::new();
    assert_eq!(empty1, empty2);
}

/// Test EmptyStorage display.
#[test]
fn test_emptystorage_display() {
    let empty: EmptyStorage<f64> = EmptyStorage::new();
    let display = format!("{}", empty);
    assert!(display.contains("EmptyStorage"));
}

/// Test EmptyStorage debug.
#[test]
fn test_emptystorage_debug() {
    let empty: EmptyStorage<f64> = EmptyStorage::new();
    let debug = format!("{:?}", empty);
    assert!(debug.contains("EmptyStorage"));
}

/// Test EmptyStorage clone.
#[test]
fn test_emptystorage_clone() {
    let empty1: EmptyStorage<f64> = EmptyStorage::new();
    let empty2 = empty1.clone();
    assert_eq!(empty1, empty2);
}

/// Test EmptyNumberStorage alias.
#[test]
fn test_emptynumberstorage_alias() {
    use ndtensors::EmptyNumberStorage;
    let empty: EmptyNumberStorage = EmptyNumberStorage::new();
    assert!(empty.is_empty());
}

/// Test EmptyStorage with complex f64 element type.
#[test]
fn test_emptystorage_complex_f64() {
    use ndtensors::c64;
    let empty: EmptyStorage<c64> = EmptyStorage::new();
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}

/// Test EmptyStorage is_empty is always true.
#[test]
fn test_emptystorage_is_empty_always_true() {
    let empty_f64: EmptyStorage<f64> = EmptyStorage::new();
    let empty_empty: EmptyStorage<EmptyNumber> = EmptyStorage::new();

    assert!(empty_f64.is_empty());
    assert!(empty_empty.is_empty());
}

/// Test EmptyStorage len is always 0.
#[test]
fn test_emptystorage_len_always_zero() {
    let empty_f64: EmptyStorage<f64> = EmptyStorage::new();
    let empty_empty: EmptyStorage<EmptyNumber> = EmptyStorage::new();

    assert_eq!(empty_f64.len(), 0);
    assert_eq!(empty_empty.len(), 0);
}

/// Test EmptyStorage::empty() constructor for EmptyNumber.
#[test]
fn test_emptystorage_empty_constructor() {
    let empty = EmptyStorage::<EmptyNumber>::empty();
    assert!(empty.is_empty());
}
