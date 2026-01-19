//! Outer product operation for tensors.
//!
//! This module provides the outer product (tensor product) of two tensors.

use std::ops::Mul;

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::storage::Dense;
use crate::tensor::Tensor;

/// Compute the outer product of two tensors.
///
/// For A with shape [a0, a1, ...] and B with shape [b0, b1, ...],
/// returns C with shape [a0, a1, ..., b0, b1, ...] where
/// C[i0, i1, ..., j0, j1, ...] = A[i0, i1, ...] * B[j0, j1, ...]
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::outer;
///
/// let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
/// let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();
/// let c = outer(&a, &b).unwrap();
///
/// assert_eq!(c.shape(), &[2, 3]);
/// // c = [[3, 4, 5], [6, 8, 10]] in column-major order
/// assert_eq!(c.get(&[0, 0]), Some(&3.0));
/// assert_eq!(c.get(&[1, 0]), Some(&6.0));
/// assert_eq!(c.get(&[0, 1]), Some(&4.0));
/// assert_eq!(c.get(&[1, 1]), Some(&8.0));
/// ```
pub fn outer<ElT: Scalar + Mul<Output = ElT>>(
    a: &Tensor<ElT, Dense<ElT>>,
    b: &Tensor<ElT, Dense<ElT>>,
) -> Result<Tensor<ElT, Dense<ElT>>, TensorError> {
    // Compute output shape: [a_shape..., b_shape...]
    let mut output_shape: Vec<usize> = a.shape().to_vec();
    output_shape.extend(b.shape());

    // Handle empty shapes (scalar tensors)
    if output_shape.is_empty() {
        output_shape.push(1);
    }

    let mut result = Tensor::<ElT, Dense<ElT>>::zeros(&output_shape);

    outer_into(&mut result, a, b)?;

    Ok(result)
}

/// In-place outer product into pre-allocated result tensor.
///
/// The result tensor must have shape [a_shape..., b_shape...].
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::{outer, outer_into};
///
/// let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
/// let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();
/// let mut c = Tensor::<f64>::zeros(&[2, 3]);
///
/// outer_into(&mut c, &a, &b).unwrap();
///
/// assert_eq!(c.get(&[0, 0]), Some(&3.0));
/// assert_eq!(c.get(&[1, 0]), Some(&6.0));
/// ```
pub fn outer_into<ElT: Scalar + Mul<Output = ElT>>(
    result: &mut Tensor<ElT, Dense<ElT>>,
    a: &Tensor<ElT, Dense<ElT>>,
    b: &Tensor<ElT, Dense<ElT>>,
) -> Result<(), TensorError> {
    let a_len = a.len();
    let b_len = b.len();
    let expected_len = a_len * b_len;

    if result.len() != expected_len {
        return Err(TensorError::ShapeMismatch {
            expected: expected_len,
            actual: result.len(),
        });
    }

    // Get slices
    let a_data = a.data();
    let b_data = b.data();
    let result_data = result.data_mut();

    // Compute outer product: C[i, j] = A[i] * B[j]
    // In column-major order: linear index = i + j * a_len
    for j in 0..b_len {
        let b_j = b_data[j];
        for i in 0..a_len {
            result_data[i + j * a_len] = a_data[i] * b_j;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::c64;

    #[test]
    fn test_outer_1d_1d() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();
        let c = outer(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);

        // c[i, j] = a[i] * b[j]
        assert_eq!(*c.get(&[0, 0]).unwrap(), 3.0); // 1*3
        assert_eq!(*c.get(&[1, 0]).unwrap(), 6.0); // 2*3
        assert_eq!(*c.get(&[0, 1]).unwrap(), 4.0); // 1*4
        assert_eq!(*c.get(&[1, 1]).unwrap(), 8.0); // 2*4
        assert_eq!(*c.get(&[0, 2]).unwrap(), 5.0); // 1*5
        assert_eq!(*c.get(&[1, 2]).unwrap(), 10.0); // 2*5
    }

    #[test]
    fn test_outer_2d_1d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0], &[2]).unwrap();
        let c = outer(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2, 2]);

        // c[i, j, k] = a[i, j] * b[k]
        // a is column-major: a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4
        assert_eq!(*c.get(&[0, 0, 0]).unwrap(), 10.0); // 1*10
        assert_eq!(*c.get(&[1, 0, 0]).unwrap(), 20.0); // 2*10
        assert_eq!(*c.get(&[0, 1, 0]).unwrap(), 30.0); // 3*10
        assert_eq!(*c.get(&[1, 1, 0]).unwrap(), 40.0); // 4*10
        assert_eq!(*c.get(&[0, 0, 1]).unwrap(), 20.0); // 1*20
        assert_eq!(*c.get(&[1, 0, 1]).unwrap(), 40.0); // 2*20
    }

    #[test]
    fn test_outer_1d_2d() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let c = outer(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2, 2]);

        // c[i, j, k] = a[i] * b[j, k]
        // b is column-major: b[0,0]=10, b[1,0]=20, b[0,1]=30, b[1,1]=40
        assert_eq!(*c.get(&[0, 0, 0]).unwrap(), 10.0); // 1*10
        assert_eq!(*c.get(&[1, 0, 0]).unwrap(), 20.0); // 2*10
        assert_eq!(*c.get(&[0, 1, 0]).unwrap(), 20.0); // 1*20
        assert_eq!(*c.get(&[1, 1, 0]).unwrap(), 40.0); // 2*20
    }

    #[test]
    fn test_outer_scalar() {
        // Scalar (0D) times 1D
        let a = Tensor::from_vec(vec![2.0], &[]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();
        let c = outer(&a, &b).unwrap();

        assert_eq!(c.shape(), &[3]);
        assert_eq!(*c.get(&[0]).unwrap(), 6.0); // 2*3
        assert_eq!(*c.get(&[1]).unwrap(), 8.0); // 2*4
        assert_eq!(*c.get(&[2]).unwrap(), 10.0); // 2*5
    }

    #[test]
    fn test_outer_complex() {
        let a = Tensor::from_vec(vec![c64::new(1.0, 1.0), c64::new(2.0, 0.0)], &[2]).unwrap();
        let b = Tensor::from_vec(vec![c64::new(1.0, 0.0), c64::new(0.0, 1.0)], &[2]).unwrap();
        let c = outer(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);

        // c[0,0] = (1+i) * 1 = 1+i
        let c00 = c.get(&[0, 0]).unwrap();
        assert_eq!(c00.re, 1.0);
        assert_eq!(c00.im, 1.0);

        // c[0,1] = (1+i) * i = i + i^2 = -1+i
        let c01 = c.get(&[0, 1]).unwrap();
        assert_eq!(c01.re, -1.0);
        assert_eq!(c01.im, 1.0);
    }

    #[test]
    fn test_outer_into() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();
        let mut c = Tensor::<f64>::zeros(&[2, 3]);

        outer_into(&mut c, &a, &b).unwrap();

        assert_eq!(*c.get(&[0, 0]).unwrap(), 3.0);
        assert_eq!(*c.get(&[1, 0]).unwrap(), 6.0);
    }

    #[test]
    fn test_outer_into_wrong_size() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();
        let mut c = Tensor::<f64>::zeros(&[3, 3]); // Wrong size

        assert!(outer_into(&mut c, &a, &b).is_err());
    }
}
