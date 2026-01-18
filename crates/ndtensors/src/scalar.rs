//! Scalar trait for tensor element types.

use faer_traits::ComplexField;
use std::fmt::Debug;

pub use faer::c64;

/// Trait for scalar types supported by ndtensors.
///
/// This trait wraps faer's `ComplexField` with additional bounds
/// required for tensor operations.
pub trait Scalar: ComplexField + Copy + Debug + Default + 'static {
    /// The real type associated with this scalar.
    type Real: Scalar;

    /// Returns the additive identity (zero).
    fn zero() -> Self {
        Self::default()
    }

    /// Returns the multiplicative identity (one).
    fn one() -> Self;
}

impl Scalar for f64 {
    type Real = f64;

    fn one() -> Self {
        1.0
    }
}

impl Scalar for c64 {
    type Real = f64;

    fn one() -> Self {
        c64::new(1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer_traits::ComplexField;

    #[test]
    fn test_f64_is_real() {
        assert!(<f64 as ComplexField>::IS_REAL);
    }

    #[test]
    fn test_c64_is_not_real() {
        assert!(!<c64 as ComplexField>::IS_REAL);
    }

    #[test]
    fn test_c64_creation() {
        let z = c64::new(1.0, 2.0);
        assert_eq!(z.re, 1.0);
        assert_eq!(z.im, 2.0);
    }

    #[test]
    fn test_zero_one() {
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert_eq!(c64::zero(), c64::new(0.0, 0.0));
        assert_eq!(c64::one(), c64::new(1.0, 0.0));
    }
}
