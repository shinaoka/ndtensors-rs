//! Tensor conversion operations.

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Convert a 2D tensor to a nested Vec<Vec<T>>.
///
/// Returns data in row-major format (outer vec is rows, inner vec is columns).
///
/// # Errors
///
/// Returns error if the tensor is not 2D.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::to_nested_vec_2d;
///
/// // Column-major storage: data = [1, 2, 3, 4, 5, 6] for shape [2, 3]
/// // represents [[1, 3, 5], [2, 4, 6]] (each column stored contiguously)
/// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
/// let nested = to_nested_vec_2d(&t).unwrap();
/// assert_eq!(nested.len(), 2);  // 2 rows
/// assert_eq!(nested[0].len(), 3);  // 3 columns
/// assert_eq!(nested[0], vec![1.0, 3.0, 5.0]);  // row 0
/// assert_eq!(nested[1], vec![2.0, 4.0, 6.0]);  // row 1
/// ```
pub fn to_nested_vec_2d<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
) -> Result<Vec<Vec<ElT>>, TensorError> {
    if tensor.ndim() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: tensor.ndim(),
        });
    }

    let shape = tensor.shape();
    let nrows = shape[0];
    let ncols = shape[1];

    let mut result = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let mut row = Vec::with_capacity(ncols);
        for j in 0..ncols {
            row.push(*tensor.get(&[i, j]).unwrap());
        }
        result.push(row);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c64;

    #[test]
    fn test_to_nested_vec_2d() {
        // Column-major 2x3 matrix
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let nested = to_nested_vec_2d(&t).unwrap();

        assert_eq!(nested.len(), 2);
        assert_eq!(nested[0].len(), 3);
        assert_eq!(nested[0], vec![1.0, 3.0, 5.0]);
        assert_eq!(nested[1], vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_to_nested_vec_2d_square() {
        // Column-major 3x3 matrix
        let t = DenseTensor::from_vec(vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0], &[3, 3])
            .unwrap();
        let nested = to_nested_vec_2d(&t).unwrap();

        assert_eq!(nested.len(), 3);
        assert_eq!(nested[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(nested[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(nested[2], vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_to_nested_vec_2d_not_2d() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let result = to_nested_vec_2d(&t);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_nested_vec_2d_c64() {
        let t = DenseTensor::from_vec(
            vec![
                c64::new(1.0, 1.0),
                c64::new(2.0, 2.0),
                c64::new(3.0, 3.0),
                c64::new(4.0, 4.0),
            ],
            &[2, 2],
        )
        .unwrap();
        let nested = to_nested_vec_2d(&t).unwrap();

        assert_eq!(nested.len(), 2);
        assert_eq!(nested[0][0].re, 1.0);
        assert_eq!(nested[0][1].re, 3.0);
        assert_eq!(nested[1][0].re, 2.0);
        assert_eq!(nested[1][1].re, 4.0);
    }
}
