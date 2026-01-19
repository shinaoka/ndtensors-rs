//! Scalar trait for tensor element types.

use faer_traits::ComplexField;
use std::fmt::Debug;

pub use faer::c64;

/// Marker trait for real scalar types with additional operations.
pub trait RealScalar: Scalar + std::ops::Add<Output = Self> {
    /// Square root.
    fn sqrt(self) -> Self;

    /// Exponential function.
    fn exp(self) -> Self;

    /// Convert to f64.
    fn to_f64(self) -> f64;
}

impl RealScalar for f64 {
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn to_f64(self) -> f64 {
        self
    }
}

/// Trait for scalar types supported by ndtensors.
///
/// This trait wraps faer's `ComplexField` with additional bounds
/// required for tensor operations.
pub trait Scalar: ComplexField + Copy + Debug + Default + 'static {
    /// The real type associated with this scalar.
    type Real: RealScalar;

    /// Returns the additive identity (zero).
    fn zero() -> Self {
        Self::default()
    }

    /// Returns the multiplicative identity (one).
    fn one() -> Self;

    /// Returns the real part of this scalar.
    fn real_part(&self) -> <Self as Scalar>::Real;

    /// Returns the imaginary part of this scalar.
    fn imag_part(&self) -> <Self as Scalar>::Real;

    /// Returns the complex conjugate of this scalar.
    fn conjugate(&self) -> Self;

    /// Returns the squared absolute value: |x|^2.
    fn abs_sqr(&self) -> <Self as Scalar>::Real;
}

impl Scalar for f64 {
    type Real = f64;

    fn one() -> Self {
        1.0
    }

    fn real_part(&self) -> <Self as Scalar>::Real {
        *self
    }

    fn imag_part(&self) -> <Self as Scalar>::Real {
        0.0
    }

    fn conjugate(&self) -> Self {
        *self
    }

    fn abs_sqr(&self) -> <Self as Scalar>::Real {
        self * self
    }
}

impl Scalar for c64 {
    type Real = f64;

    fn one() -> Self {
        c64::new(1.0, 0.0)
    }

    fn real_part(&self) -> <Self as Scalar>::Real {
        self.re
    }

    fn imag_part(&self) -> <Self as Scalar>::Real {
        self.im
    }

    fn conjugate(&self) -> Self {
        c64::new(self.re, -self.im)
    }

    fn abs_sqr(&self) -> <Self as Scalar>::Real {
        self.re * self.re + self.im * self.im
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_real_imag_f64() {
        let x: f64 = 3.5;
        assert_eq!(x.real_part(), 3.5);
        assert_eq!(x.imag_part(), 0.0);
    }

    #[test]
    fn test_real_imag_c64() {
        let z = c64::new(3.0, 4.0);
        assert_eq!(z.real_part(), 3.0);
        assert_eq!(z.imag_part(), 4.0);
    }

    #[test]
    fn test_conjugate_f64() {
        let x: f64 = 3.5;
        assert_eq!(x.conjugate(), 3.5);
    }

    #[test]
    fn test_conjugate_c64() {
        let z = c64::new(3.0, 4.0);
        let conj = z.conjugate();
        assert_eq!(conj.re, 3.0);
        assert_eq!(conj.im, -4.0);
    }

    #[test]
    fn test_abs_sqr_f64() {
        let x: f64 = 3.0;
        assert_eq!(x.abs_sqr(), 9.0);
    }

    #[test]
    fn test_abs_sqr_c64() {
        let z = c64::new(3.0, 4.0);
        assert_eq!(z.abs_sqr(), 25.0); // 3^2 + 4^2 = 9 + 16 = 25
    }
}
