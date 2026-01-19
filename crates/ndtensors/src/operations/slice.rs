//! Tensor slicing operations.

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;
use std::ops::Range;

/// Extract a slice from a tensor along specified dimensions.
///
/// This creates a copy of the sliced data (not a view).
///
/// # Arguments
///
/// * `tensor` - Source tensor
/// * `ranges` - Slice specification for each dimension (start..end)
///
/// # Errors
///
/// Returns error if:
/// - Number of ranges doesn't match tensor dimensions
/// - Any range is out of bounds
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::slice;
///
/// let t = Tensor::<f64>::ones(&[4, 5, 6]);
/// let s = slice(&t, &[1..3, 0..5, 2..4]).unwrap();
/// assert_eq!(s.shape(), &[2, 5, 2]);
/// ```
pub fn slice<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    ranges: &[Range<usize>],
) -> Result<DenseTensor<ElT>, TensorError> {
    let shape = tensor.shape();
    let ndim = tensor.ndim();

    // Validate number of ranges
    if ranges.len() != ndim {
        return Err(TensorError::WrongNumberOfIndices {
            expected: ndim,
            actual: ranges.len(),
        });
    }

    // Validate ranges and compute new shape
    let mut new_shape = Vec::with_capacity(ndim);
    for (dim, range) in ranges.iter().enumerate() {
        if range.start > range.end || range.end > shape[dim] {
            return Err(TensorError::SliceOutOfBounds {
                start: range.start,
                end: range.end,
                dim,
                size: shape[dim],
            });
        }
        new_shape.push(range.end - range.start);
    }

    // Compute total size
    let total_size: usize = new_shape.iter().product();
    let mut data = Vec::with_capacity(total_size);

    // Copy data using multi-dimensional iteration
    let mut indices = vec![0usize; ndim];
    for _ in 0..total_size {
        // Compute source indices
        let src_indices: Vec<usize> = indices
            .iter()
            .zip(ranges.iter())
            .map(|(&i, range)| range.start + i)
            .collect();
        data.push(*tensor.get(&src_indices).unwrap());

        // Increment indices (column-major order)
        for d in 0..ndim {
            indices[d] += 1;
            if indices[d] < new_shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }

    DenseTensor::from_vec(data, &new_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c64;

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn test_slice_1d() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let s = slice(&t, &[1..4]).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_2d() {
        // Column-major 3x4 matrix
        // [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let t = DenseTensor::from_vec(data, &[3, 4]).unwrap();

        // Slice rows 0..2, cols 1..3
        let s = slice(&t, &[0..2, 1..3]).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        // Expected: [[4, 7], [5, 8]] in column-major = [4, 5, 7, 8]
        assert_eq!(s.data(), &[4.0, 5.0, 7.0, 8.0]);
    }

    #[test]
    fn test_slice_3d() {
        let t = DenseTensor::<f64>::ones(&[4, 5, 6]);
        let s = slice(&t, &[1..3, 2..4, 0..2]).unwrap();
        assert_eq!(s.shape(), &[2, 2, 2]);
        assert_eq!(s.len(), 8);
    }

    #[test]
    fn test_slice_full() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let s = slice(&t, &[0..2, 0..2]).unwrap();
        assert_eq!(s.shape(), t.shape());
        assert_eq!(s.data(), t.data());
    }

    #[test]
    fn test_slice_single_element() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let s = slice(&t, &[0..1, 1..2]).unwrap();
        assert_eq!(s.shape(), &[1, 1]);
        assert_eq!(s.data(), &[3.0]); // element at (0, 1) in column-major
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn test_slice_wrong_num_ranges() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let result = slice(&t, &[0..2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_out_of_bounds() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let result = slice(&t, &[0..3, 0..2]);
        assert!(result.is_err());
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn test_slice_invalid_range() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let result = slice(&t, &[1..0, 0..2]); // start > end
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_c64() {
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
        let s = slice(&t, &[0..1, 0..2]).unwrap();
        assert_eq!(s.shape(), &[1, 2]);
        assert_eq!(s.get(&[0, 0]).unwrap().re, 1.0);
        assert_eq!(s.get(&[0, 1]).unwrap().re, 3.0);
    }
}
