//! Tensor norm operations.

use crate::scalar::{RealScalar, Scalar};
use crate::tensor::DenseTensor;

/// Compute the Frobenius norm (L2 norm) of a tensor.
///
/// For a tensor T, returns sqrt(sum(|T_i|^2)) where the sum is over all elements.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::norm;
///
/// let t = Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap();
/// assert!((norm(&t) - 5.0).abs() < 1e-10);
/// ```
pub fn norm<ElT: Scalar>(tensor: &DenseTensor<ElT>) -> <ElT as Scalar>::Real {
    RealScalar::sqrt(norm_sqr(tensor))
}

/// Compute the squared Frobenius norm of a tensor.
///
/// More efficient than `norm` when the square root is not needed.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::norm_sqr;
///
/// let t = Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap();
/// assert!((norm_sqr(&t) - 25.0).abs() < 1e-10);
/// ```
pub fn norm_sqr<ElT: Scalar>(tensor: &DenseTensor<ElT>) -> <ElT as Scalar>::Real {
    let data = tensor.data();
    let mut sum = <ElT as Scalar>::Real::zero();
    for &x in data {
        sum = sum + x.abs_sqr();
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c64;
    use approx::assert_relative_eq;

    #[test]
    fn test_norm_f64_1d() {
        // Simple 1D case: 3-4-5 triangle
        let t = DenseTensor::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        assert_relative_eq!(norm(&t), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_norm_f64_2d() {
        // 2D case
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        // sqrt(1 + 4 + 9 + 16) = sqrt(30)
        assert_relative_eq!(norm(&t), 30.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_norm_c64() {
        // For complex numbers, |z|^2 = re^2 + im^2
        let t = DenseTensor::from_vec(vec![c64::new(3.0, 4.0)], &[1]).unwrap();
        // |3+4i| = 5, so norm = 5
        assert_relative_eq!(norm(&t), 5.0, epsilon = 1e-10);

        let t2 = DenseTensor::from_vec(vec![c64::new(1.0, 0.0), c64::new(0.0, 1.0)], &[2]).unwrap();
        // |1|^2 + |i|^2 = 1 + 1 = 2, norm = sqrt(2)
        assert_relative_eq!(norm(&t2), 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_norm_sqr() {
        let t = DenseTensor::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        assert_relative_eq!(norm_sqr(&t), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_norm_scalar() {
        // Scalar (0-dimensional) tensor
        let t = DenseTensor::<f64>::zeros(&[]);
        assert_relative_eq!(norm(&t), 0.0, epsilon = 1e-10);

        let t2 = DenseTensor::from_vec(vec![5.0], &[]).unwrap();
        assert_relative_eq!(norm(&t2), 5.0, epsilon = 1e-10);
    }
}
