//! Tests for EmptyNumber.
//!
//! These tests mirror NDTensors.jl's test_emptynumber.jl, covering:
//! - Type promotion
//! - Basic arithmetic (add, subtract, multiply, divide)
//! - Conversion (complex, float)
//! - Norm computation

use ndtensors::EmptyNumber;

/// Test EmptyNumber type equality.
#[test]
fn test_emptynumber_equality() {
    let e1 = EmptyNumber;
    let e2 = EmptyNumber;
    assert_eq!(e1, e2);
}

/// Test EmptyNumber default.
#[test]
fn test_emptynumber_default() {
    let e: EmptyNumber = Default::default();
    assert_eq!(e, EmptyNumber);
}

/// Test EmptyNumber + EmptyNumber = EmptyNumber.
/// Mirrors: @test 𝟎 + 𝟎 == 𝟎
#[test]
fn test_emptynumber_add_empty() {
    let e = EmptyNumber;
    assert_eq!(e + e, EmptyNumber);
}

/// Test EmptyNumber + x = x.
/// Mirrors: @test 𝟎 + x == x
#[test]
fn test_emptynumber_add_f64() {
    let e = EmptyNumber;
    let x = 2.3f64;
    assert_eq!(e + x, x);
}

/// Test x + EmptyNumber = x.
/// Mirrors: @test x + 𝟎 == x
#[test]
fn test_f64_add_emptynumber() {
    let e = EmptyNumber;
    let x = 2.3f64;
    assert_eq!(x + e, x);
}

/// Test -EmptyNumber = EmptyNumber.
/// Mirrors: @test -𝟎 == 𝟎
#[test]
fn test_emptynumber_neg() {
    let e = EmptyNumber;
    assert_eq!(-e, EmptyNumber);
}

/// Test EmptyNumber - EmptyNumber = EmptyNumber.
/// Mirrors: @test 𝟎 - 𝟎 == 𝟎
#[test]
fn test_emptynumber_sub_empty() {
    let e = EmptyNumber;
    assert_eq!(e - e, EmptyNumber);
}

/// Test x - EmptyNumber = x.
/// Mirrors: @test x - 𝟎 == x
#[test]
fn test_f64_sub_emptynumber() {
    let e = EmptyNumber;
    let x = 2.3f64;
    assert_eq!(x - e, x);
}

/// Test EmptyNumber * EmptyNumber = EmptyNumber.
/// Mirrors: @test 𝟎 * 𝟎 == 𝟎
#[test]
fn test_emptynumber_mul_empty() {
    let e = EmptyNumber;
    assert_eq!(e * e, EmptyNumber);
}

/// Test x * EmptyNumber = EmptyNumber.
/// Mirrors: @test x * 𝟎 == 𝟎
#[test]
fn test_f64_mul_emptynumber() {
    let e = EmptyNumber;
    let x = 2.3f64;
    assert_eq!(x * e, EmptyNumber);
}

/// Test EmptyNumber * x = EmptyNumber.
/// Mirrors: @test 𝟎 * x == 𝟎
#[test]
fn test_emptynumber_mul_f64() {
    let e = EmptyNumber;
    let x = 2.3f64;
    assert_eq!(e * x, EmptyNumber);
}

/// Test EmptyNumber / x = EmptyNumber.
/// Mirrors: @test 𝟎 / x == 𝟎
#[test]
fn test_emptynumber_div_f64() {
    let e = EmptyNumber;
    let x = 2.3f64;
    assert_eq!(e / x, EmptyNumber);
}

/// Test x / EmptyNumber throws error.
/// Mirrors: @test_throws DivideError() x / 𝟎 == 𝟎
#[test]
#[should_panic(expected = "division by EmptyNumber")]
fn test_f64_div_emptynumber_panics() {
    let e = EmptyNumber;
    let x = 2.3f64;
    let _ = x / e;
}

/// Test EmptyNumber / EmptyNumber throws error.
/// Mirrors: @test_throws DivideError() 𝟎 / 𝟎 == 𝟎
#[test]
#[should_panic(expected = "division by EmptyNumber")]
fn test_emptynumber_div_emptynumber_panics() {
    let e = EmptyNumber;
    let _ = e / e;
}

/// Test float(EmptyNumber) == 0.0.
/// Mirrors: @test float(𝟎) == 0.0
#[test]
fn test_emptynumber_to_float() {
    let e = EmptyNumber;
    assert_eq!(f64::from(e), 0.0);
}

/// Test float(EmptyNumber) is f64.
/// Mirrors: @test float(𝟎) isa Float64
#[test]
fn test_emptynumber_to_float_type() {
    let e = EmptyNumber;
    let f: f64 = e.into();
    assert_eq!(f, 0.0f64);
}

/// Test norm(EmptyNumber) == 0.0.
/// Mirrors: @test norm(𝟎) == 0.0
#[test]
fn test_emptynumber_norm() {
    let e = EmptyNumber;
    assert_eq!(e.norm(), 0.0);
}

/// Test norm(EmptyNumber) is f64.
/// Mirrors: @test norm(𝟎) isa Float64
#[test]
fn test_emptynumber_norm_type() {
    let e = EmptyNumber;
    let n = e.norm();
    assert_eq!(n, 0.0f64);
}

/// Test EmptyNumber to_f64 method.
#[test]
fn test_emptynumber_to_f64_method() {
    let e = EmptyNumber;
    assert_eq!(e.to_f64(), 0.0);
}

/// Test EmptyNumber is_empty method.
#[test]
fn test_emptynumber_is_empty() {
    let e = EmptyNumber;
    assert!(e.is_empty());
}

/// Test EmptyNumber display.
#[test]
fn test_emptynumber_display() {
    let e = EmptyNumber;
    let display = format!("{}", e);
    assert_eq!(display, "EmptyNumber");
}

/// Test EmptyNumber debug.
#[test]
fn test_emptynumber_debug() {
    let e = EmptyNumber;
    let debug = format!("{:?}", e);
    assert_eq!(debug, "EmptyNumber");
}

/// Test EmptyNumber clone.
#[test]
fn test_emptynumber_clone() {
    let e1 = EmptyNumber;
    let e2 = e1;
    assert_eq!(e1, e2);
}

/// Test EmptyNumber copy.
#[test]
fn test_emptynumber_copy() {
    let e1 = EmptyNumber;
    let e2 = e1;
    assert_eq!(e1, e2);
    assert_eq!(e2, EmptyNumber);
}

/// Test EmptyNumber hash.
#[test]
fn test_emptynumber_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(EmptyNumber);
    set.insert(EmptyNumber);
    // Both should be the same
    assert_eq!(set.len(), 1);
}
