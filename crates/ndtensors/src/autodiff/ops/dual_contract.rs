//! Dual tensor contraction with JVP (forward-mode AD).

use crate::autodiff::dual::DualTensor;
use crate::contract::contract;
use crate::error::TensorError;
use crate::operations::apply_binary;
use crate::scalar::Scalar;
use std::ops::{Add, Mul};

/// Compute tensor contraction with JVP (forward-mode AD).
///
/// For contraction `C = contract(A, B)`, the JVP is computed using the
/// Leibniz product rule:
///
///   dC = contract(dA, B) + contract(A, dB)
///
/// where dA and dB are the tangent vectors of A and B.
///
/// # Arguments
///
/// * `a` - First dual tensor
/// * `labels_a` - Labels for each dimension of `a` (negative = contracted)
/// * `b` - Second dual tensor
/// * `labels_b` - Labels for each dimension of `b` (negative = contracted)
///
/// # Returns
///
/// A dual tensor containing:
/// - primal: contract(a.primal, b.primal)
/// - tangent: contract(a.tangent, b.primal) + contract(a.primal, b.tangent)
///
/// # Optimization
///
/// - If both tangents are None (zero), the output tangent is None
/// - If only one tangent is None, only one contraction is computed for the tangent
///
/// # Example
///
/// ```ignore
/// use ndtensors::autodiff::{DualTensor, dual_contract};
/// use ndtensors::Tensor;
///
/// let a = DualTensor::with_tangent(
///     Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap(),
///     Tensor::ones(&[2, 3]), // dA = ones
/// ).unwrap();
/// let b = DualTensor::new(Tensor::ones(&[3, 4])); // dB = 0
///
/// // C = A @ B with JVP
/// let c = dual_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
///
/// // c.tangent() = dA @ B (since dB = 0)
/// assert!(c.has_tangent());
/// ```
pub fn dual_contract<T: Scalar + Add<Output = T> + Mul<Output = T>>(
    a: &DualTensor<T>,
    labels_a: &[i32],
    b: &DualTensor<T>,
    labels_b: &[i32],
) -> Result<DualTensor<T>, TensorError> {
    // Compute primal: C = contract(A, B)
    let primal = contract(a.primal(), labels_a, b.primal(), labels_b)?;

    // Compute tangent using Leibniz rule: dC = contract(dA, B) + contract(A, dB)
    let tangent = match (a.tangent(), b.tangent()) {
        (None, None) => {
            // Both inputs are constants, output tangent is zero
            None
        }
        (Some(da), None) => {
            // dC = contract(dA, B)
            let dc = contract(da, labels_a, b.primal(), labels_b)?;
            Some(dc)
        }
        (None, Some(db)) => {
            // dC = contract(A, dB)
            let dc = contract(a.primal(), labels_a, db, labels_b)?;
            Some(dc)
        }
        (Some(da), Some(db)) => {
            // dC = contract(dA, B) + contract(A, dB)
            let dc1 = contract(da, labels_a, b.primal(), labels_b)?;
            let dc2 = contract(a.primal(), labels_a, db, labels_b)?;
            let dc = apply_binary(&dc1, &dc2, |x, y| x + y)?;
            Some(dc)
        }
    };

    Ok(DualTensor::from_primal_tangent(primal, tangent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use approx::assert_relative_eq;

    #[test]
    fn test_dual_contract_no_tangent() {
        // Both inputs are constants
        let a: DualTensor<f64> = DualTensor::new(Tensor::ones(&[2, 3]));
        let b: DualTensor<f64> = DualTensor::new(Tensor::ones(&[3, 4]));

        let c = dual_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert_eq!(c.shape(), &[2, 4]);
        assert!(!c.has_tangent());
        // Primal: ones(2,3) @ ones(3,4) = 3 * ones(2,4)
        assert_relative_eq!(*c.primal().get(&[0, 0]).unwrap(), 3.0);
    }

    #[test]
    fn test_dual_contract_one_tangent_a() {
        // Only A has a tangent
        let a: DualTensor<f64> =
            DualTensor::with_tangent(Tensor::ones(&[2, 3]), Tensor::ones(&[2, 3])).unwrap();
        let b: DualTensor<f64> = DualTensor::new(Tensor::ones(&[3, 4]));

        let c = dual_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert!(c.has_tangent());
        assert_eq!(c.tangent().unwrap().shape(), &[2, 4]);

        // dC = dA @ B = ones(2,3) @ ones(3,4) = 3 * ones(2,4)
        assert_relative_eq!(*c.tangent().unwrap().get(&[0, 0]).unwrap(), 3.0);
        assert_relative_eq!(*c.tangent().unwrap().get(&[1, 3]).unwrap(), 3.0);
    }

    #[test]
    fn test_dual_contract_one_tangent_b() {
        // Only B has a tangent
        let a: DualTensor<f64> = DualTensor::new(Tensor::ones(&[2, 3]));
        let b: DualTensor<f64> =
            DualTensor::with_tangent(Tensor::ones(&[3, 4]), Tensor::ones(&[3, 4])).unwrap();

        let c = dual_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert!(c.has_tangent());
        // dC = A @ dB = ones(2,3) @ ones(3,4) = 3 * ones(2,4)
        assert_relative_eq!(*c.tangent().unwrap().get(&[0, 0]).unwrap(), 3.0);
    }

    #[test]
    fn test_dual_contract_both_tangents() {
        // Both inputs have tangents
        let a: DualTensor<f64> =
            DualTensor::with_tangent(Tensor::ones(&[2, 3]), Tensor::ones(&[2, 3])).unwrap();
        let b: DualTensor<f64> =
            DualTensor::with_tangent(Tensor::ones(&[3, 4]), Tensor::ones(&[3, 4])).unwrap();

        let c = dual_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert!(c.has_tangent());
        // dC = dA @ B + A @ dB = 3 + 3 = 6
        assert_relative_eq!(*c.tangent().unwrap().get(&[0, 0]).unwrap(), 6.0);
    }

    #[test]
    fn test_dual_contract_inner_product() {
        // Inner product: a . b
        let a: DualTensor<f64> = DualTensor::with_tangent(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
            Tensor::from_vec(vec![1.0, 0.0, 0.0], &[3]).unwrap(), // d/da[0]
        )
        .unwrap();
        let b: DualTensor<f64> =
            DualTensor::new(Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap());

        let c = dual_contract(&a, &[-1], &b, &[-1]).unwrap();

        // Primal: 1*4 + 2*5 + 3*6 = 32
        assert_relative_eq!(*c.primal().get_linear(0).unwrap(), 32.0);
        // Tangent: d/da[0](a.b) = b[0] = 4
        assert_relative_eq!(*c.tangent().unwrap().get_linear(0).unwrap(), 4.0);
    }

    #[test]
    fn test_dual_contract_outer_product() {
        // Outer product: C[i,j] = a[i] * b[j]
        let a: DualTensor<f64> = DualTensor::with_tangent(
            Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
            Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap(), // d/da[0]
        )
        .unwrap();
        let b: DualTensor<f64> =
            DualTensor::new(Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap());

        let c = dual_contract(&a, &[1], &b, &[2]).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        // Primal: C[0,0] = a[0] * b[0] = 1 * 3 = 3
        assert_relative_eq!(*c.primal().get(&[0, 0]).unwrap(), 3.0);
        // Tangent: dC[0,0] = da[0] * b[0] = 1 * 3 = 3
        assert_relative_eq!(*c.tangent().unwrap().get(&[0, 0]).unwrap(), 3.0);
        // Tangent: dC[1,0] = da[1] * b[0] = 0 * 3 = 0
        assert_relative_eq!(*c.tangent().unwrap().get(&[1, 0]).unwrap(), 0.0);
    }

    #[test]
    fn test_dual_contract_3d_tensor() {
        // A[i,j,k] * B[k,l] -> C[i,j,l]
        let a: DualTensor<f64> =
            DualTensor::with_tangent(Tensor::ones(&[2, 3, 4]), Tensor::ones(&[2, 3, 4])).unwrap();
        let b: DualTensor<f64> = DualTensor::new(Tensor::ones(&[4, 5]));

        let c = dual_contract(&a, &[1, 2, -1], &b, &[-1, 3]).unwrap();

        assert_eq!(c.shape(), &[2, 3, 5]);
        // Primal: each element = sum over k of 1*1 = 4
        assert_relative_eq!(*c.primal().get(&[0, 0, 0]).unwrap(), 4.0);
        // Tangent: dC = dA @ B, each element = 4
        assert_relative_eq!(*c.tangent().unwrap().get(&[0, 0, 0]).unwrap(), 4.0);
    }

    #[test]
    fn test_dual_contract_vs_finite_diff() {
        // Verify dual_contract JVP against finite differences
        let eps = 1e-7;

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let da_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        let a_primal = Tensor::<f64>::from_vec(a_data.clone(), &[2, 3]).unwrap();
        let a_tangent = Tensor::<f64>::from_vec(da_data.clone(), &[2, 3]).unwrap();
        let b = Tensor::<f64>::from_vec(b_data.clone(), &[3, 4]).unwrap();

        // Compute using dual_contract
        let dual_a: DualTensor<f64> =
            DualTensor::with_tangent(a_primal.clone(), a_tangent).unwrap();
        let dual_b: DualTensor<f64> = DualTensor::new(b.clone());
        let result = dual_contract(&dual_a, &[1, -1], &dual_b, &[-1, 2]).unwrap();
        let jvp = result.tangent().unwrap();

        // Compute finite difference
        let a_plus_data: Vec<f64> = a_data
            .iter()
            .zip(da_data.iter())
            .map(|(&x, &d)| x + eps * d)
            .collect();
        let a_plus = Tensor::<f64>::from_vec(a_plus_data, &[2, 3]).unwrap();

        let c = contract(&a_primal, &[1, -1], &b, &[-1, 2]).unwrap();
        let c_plus = contract(&a_plus, &[1, -1], &b, &[-1, 2]).unwrap();

        // Check each element
        for i in 0..c.len() {
            let fd = (*c_plus.get_linear(i).unwrap() - *c.get_linear(i).unwrap()) / eps;
            assert_relative_eq!(*jvp.get_linear(i).unwrap(), fd, epsilon = 1e-5);
        }
    }
}
