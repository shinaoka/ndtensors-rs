//! EmptyNumber type for ndtensors.
//!
//! EmptyNumber is a special number type that represents an uninitialized or empty value.
//! It can be promoted to any numeric type and acts as zero in arithmetic operations.
//! This mirrors NDTensors.jl's `EmptyNumber` type.
//!
//! # Properties
//!
//! - EmptyNumber + EmptyNumber = EmptyNumber
//! - EmptyNumber + x = x
//! - x + EmptyNumber = x
//! - EmptyNumber * x = EmptyNumber
//! - x * EmptyNumber = EmptyNumber
//! - EmptyNumber / x = EmptyNumber
//! - x / EmptyNumber = Error (DivideByZero)
//!
//! # Example
//!
//! ```
//! use ndtensors::empty_number::EmptyNumber;
//!
//! let empty = EmptyNumber;
//!
//! // Basic arithmetic
//! assert_eq!(empty + empty, empty);
//! assert_eq!(empty * empty, empty);
//! assert_eq!(-empty, empty);
//!
//! // Conversion to f64
//! assert_eq!(f64::from(empty), 0.0);
//! ```

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// EmptyNumber - a placeholder number that can be promoted to any type.
///
/// EmptyNumber acts as an "unset" value that behaves like zero in most operations
/// but can be promoted to any numeric type. This is useful for representing
/// structurally zero elements in sparse tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct EmptyNumber;

/// Error for division by EmptyNumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DivideByEmptyNumberError;

impl fmt::Display for DivideByEmptyNumberError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "division by EmptyNumber")
    }
}

impl std::error::Error for DivideByEmptyNumberError {}

impl fmt::Display for EmptyNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EmptyNumber")
    }
}

// EmptyNumber + EmptyNumber = EmptyNumber
impl Add for EmptyNumber {
    type Output = EmptyNumber;

    fn add(self, _rhs: EmptyNumber) -> EmptyNumber {
        EmptyNumber
    }
}

// EmptyNumber - EmptyNumber = EmptyNumber
impl Sub for EmptyNumber {
    type Output = EmptyNumber;

    fn sub(self, _rhs: EmptyNumber) -> EmptyNumber {
        EmptyNumber
    }
}

// EmptyNumber * EmptyNumber = EmptyNumber
impl Mul for EmptyNumber {
    type Output = EmptyNumber;

    fn mul(self, _rhs: EmptyNumber) -> EmptyNumber {
        EmptyNumber
    }
}

// -EmptyNumber = EmptyNumber
impl Neg for EmptyNumber {
    type Output = EmptyNumber;

    fn neg(self) -> EmptyNumber {
        EmptyNumber
    }
}

// EmptyNumber / EmptyNumber = Error
impl Div for EmptyNumber {
    type Output = EmptyNumber;

    fn div(self, _rhs: EmptyNumber) -> EmptyNumber {
        panic!("division by EmptyNumber")
    }
}

// Arithmetic with f64
impl Add<f64> for EmptyNumber {
    type Output = f64;

    fn add(self, rhs: f64) -> f64 {
        rhs
    }
}

impl Add<EmptyNumber> for f64 {
    type Output = f64;

    fn add(self, _rhs: EmptyNumber) -> f64 {
        self
    }
}

impl Sub<EmptyNumber> for f64 {
    type Output = f64;

    fn sub(self, _rhs: EmptyNumber) -> f64 {
        self
    }
}

impl Mul<f64> for EmptyNumber {
    type Output = EmptyNumber;

    fn mul(self, _rhs: f64) -> EmptyNumber {
        EmptyNumber
    }
}

impl Mul<EmptyNumber> for f64 {
    type Output = EmptyNumber;

    fn mul(self, _rhs: EmptyNumber) -> EmptyNumber {
        EmptyNumber
    }
}

impl Div<f64> for EmptyNumber {
    type Output = EmptyNumber;

    fn div(self, _rhs: f64) -> EmptyNumber {
        EmptyNumber
    }
}

// x / EmptyNumber = Error
impl Div<EmptyNumber> for f64 {
    type Output = f64;

    fn div(self, _rhs: EmptyNumber) -> f64 {
        panic!("division by EmptyNumber")
    }
}

// Conversion from EmptyNumber to f64
impl From<EmptyNumber> for f64 {
    fn from(_: EmptyNumber) -> f64 {
        0.0
    }
}

// Conversion from EmptyNumber to f32
impl From<EmptyNumber> for f32 {
    fn from(_: EmptyNumber) -> f32 {
        0.0
    }
}

impl EmptyNumber {
    /// Create a new EmptyNumber.
    pub const fn new() -> Self {
        EmptyNumber
    }

    /// Convert to f64 (returns 0.0).
    pub fn to_f64(self) -> f64 {
        0.0
    }

    /// Compute the norm (returns 0.0).
    pub fn norm(self) -> f64 {
        0.0
    }

    /// Check if this is EmptyNumber (always true).
    pub fn is_empty(self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emptynumber_basic() {
        let e = EmptyNumber;
        assert_eq!(e, EmptyNumber);
    }

    #[test]
    fn test_emptynumber_add() {
        let e = EmptyNumber;
        assert_eq!(e + e, EmptyNumber);
    }

    #[test]
    fn test_emptynumber_add_f64() {
        let e = EmptyNumber;
        let x = 2.3f64;
        assert_eq!(e + x, x);
        assert_eq!(x + e, x);
    }

    #[test]
    fn test_emptynumber_sub() {
        let e = EmptyNumber;
        assert_eq!(e - e, EmptyNumber);
        assert_eq!(2.3f64 - e, 2.3f64);
    }

    #[test]
    fn test_emptynumber_mul() {
        let e = EmptyNumber;
        let x = 2.3f64;
        assert_eq!(e * e, EmptyNumber);
        assert_eq!(x * e, EmptyNumber);
        assert_eq!(e * x, EmptyNumber);
    }

    #[test]
    fn test_emptynumber_neg() {
        let e = EmptyNumber;
        assert_eq!(-e, EmptyNumber);
    }

    #[test]
    fn test_emptynumber_div_by_number() {
        let e = EmptyNumber;
        let x = 2.3f64;
        assert_eq!(e / x, EmptyNumber);
    }

    #[test]
    #[should_panic(expected = "division by EmptyNumber")]
    fn test_emptynumber_div_by_empty() {
        let e = EmptyNumber;
        let _ = e / e;
    }

    #[test]
    #[should_panic(expected = "division by EmptyNumber")]
    fn test_number_div_by_empty() {
        let e = EmptyNumber;
        let x = 2.3f64;
        let _ = x / e;
    }

    #[test]
    fn test_emptynumber_to_f64() {
        let e = EmptyNumber;
        assert_eq!(e.to_f64(), 0.0);
        assert_eq!(f64::from(e), 0.0);
    }

    #[test]
    fn test_emptynumber_norm() {
        let e = EmptyNumber;
        assert_eq!(e.norm(), 0.0);
    }

    #[test]
    fn test_emptynumber_display() {
        let e = EmptyNumber;
        assert_eq!(format!("{}", e), "EmptyNumber");
    }
}
