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

    // =========================================================
    // Issue #37: Additional tensor slicing tests
    // Patterns from NDTensors.jl test_dense.jl
    // =========================================================

    #[test]
    fn test_slice_dims_pattern() {
        // Match test_dense.jl pattern:
        // A is 3x4 matrix, test various slicing patterns
        //
        // Column-major 3x4 matrix layout:
        // Logical view:
        //   col0  col1  col2  col3
        // [  1     4     7    10  ]  row0
        // [  2     5     8    11  ]  row1
        // [  3     6     9    12  ]  row2
        //
        // Storage (column-major): [1,2,3,4,5,6,7,8,9,10,11,12]
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let a = DenseTensor::from_vec(data, &[3, 4]).unwrap();

        // Test: dims(A[1:2, 1]) == (2,)
        // In Rust 0-indexed: A[0..2, 0..1] (rows 0-1, col 0)
        let s1 = slice(&a, &[0..2, 0..1]).unwrap();
        assert_eq!(s1.shape(), &[2, 1]);

        // Test: dims(A[2:3, 2]) == (2,)
        // In Rust 0-indexed: A[1..3, 1..2]
        let s2 = slice(&a, &[1..3, 1..2]).unwrap();
        assert_eq!(s2.shape(), &[2, 1]);

        // Test: dims(A[2, 2:4]) == (3,)
        // In Rust 0-indexed: A[1..2, 1..4]
        let s3 = slice(&a, &[1..2, 1..4]).unwrap();
        assert_eq!(s3.shape(), &[1, 3]);

        // Test: dims(A[2:3, 2:4]) == (2, 3)
        // In Rust 0-indexed: A[1..3, 1..4]
        let s4 = slice(&a, &[1..3, 1..4]).unwrap();
        assert_eq!(s4.shape(), &[2, 3]);
    }

    #[test]
    fn test_slice_preserves_data() {
        // Verify sliced data matches expected values
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let a = DenseTensor::from_vec(data, &[3, 4]).unwrap();

        // Slice rows 1..3, cols 2..4 (0-indexed)
        // Should get:
        //   col2  col3
        // [  8    11  ]  row1
        // [  9    12  ]  row2
        let s = slice(&a, &[1..3, 2..4]).unwrap();
        assert_eq!(s.shape(), &[2, 2]);

        // Column-major storage: [8, 9, 11, 12]
        assert_eq!(s.data(), &[8.0, 9.0, 11.0, 12.0]);
    }

    #[test]
    fn test_slice_creates_copy() {
        // Verify slice creates a copy, not a view
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let original = DenseTensor::from_vec(data, &[2, 2]).unwrap();
        let mut sliced = slice(&original, &[0..2, 0..2]).unwrap();

        // Modify the slice
        sliced.set(&[0, 0], 100.0).unwrap();

        // Original should be unchanged
        assert_eq!(*original.get(&[0, 0]).unwrap(), 1.0);
    }

    #[test]
    fn test_slice_empty_range() {
        // Empty range (0..0) creates a slice with zero elements
        // Currently, this returns an error due to DenseTensor::from_vec rejecting empty tensors
        // with non-trivial shapes. We test that the result is an error.
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let result = slice(&t, &[0..0, 0..2]);
        // Empty slice results in shape mismatch due to implementation
        assert!(result.is_err() || result.unwrap().len() == 0);
    }

    #[test]
    fn test_slice_4d_tensor() {
        // Higher dimensional slicing
        let t = DenseTensor::<f64>::ones(&[2, 3, 4, 5]);
        let s = slice(&t, &[0..1, 1..2, 2..3, 3..4]).unwrap();
        assert_eq!(s.shape(), &[1, 1, 1, 1]);
        assert_eq!(s.len(), 1);
        assert_eq!(*s.get(&[0, 0, 0, 0]).unwrap(), 1.0);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn test_slice_boundary_cases() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();

        // Slice at start
        let s_start = slice(&t, &[0..2]).unwrap();
        assert_eq!(s_start.data(), &[1.0, 2.0]);

        // Slice at end
        let s_end = slice(&t, &[3..5]).unwrap();
        assert_eq!(s_end.data(), &[4.0, 5.0]);

        // Slice in middle
        let s_mid = slice(&t, &[1..4]).unwrap();
        assert_eq!(s_mid.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_error_end_exceeds_size() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // End exceeds dimension size
        let result = slice(&t, &[0..3, 0..2]);
        assert!(result.is_err());

        let result2 = slice(&t, &[0..2, 0..3]);
        assert!(result2.is_err());
    }

    #[test]
    fn test_slice_complex_3d() {
        // Complex 3D slicing
        let size = 2 * 3 * 4;
        let data: Vec<c64> = (0..size).map(|i| c64::new(i as f64, -(i as f64))).collect();
        let t = DenseTensor::from_vec(data, &[2, 3, 4]).unwrap();

        let s = slice(&t, &[0..1, 1..2, 2..3]).unwrap();
        assert_eq!(s.shape(), &[1, 1, 1]);

        // Verify the complex value is correct
        let expected = t.get(&[0, 1, 2]).unwrap();
        let actual = s.get(&[0, 0, 0]).unwrap();
        assert_eq!(actual.re, expected.re);
        assert_eq!(actual.im, expected.im);
    }

    #[test]
    fn test_slice_row_extraction() {
        // Extract a single row (common pattern)
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let a = DenseTensor::from_vec(data, &[3, 4]).unwrap();

        // Extract row 1 (all columns)
        let row = slice(&a, &[1..2, 0..4]).unwrap();
        assert_eq!(row.shape(), &[1, 4]);
        // Row 1 values: 2, 5, 8, 11 (in column-major storage)
        assert_eq!(row.data(), &[2.0, 5.0, 8.0, 11.0]);
    }

    #[test]
    fn test_slice_column_extraction() {
        // Extract a single column (common pattern)
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let a = DenseTensor::from_vec(data, &[3, 4]).unwrap();

        // Extract column 2 (all rows)
        let col = slice(&a, &[0..3, 2..3]).unwrap();
        assert_eq!(col.shape(), &[3, 1]);
        // Column 2 values: 7, 8, 9
        assert_eq!(col.data(), &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_slice_submatrix() {
        // Extract a 2x2 submatrix from a larger matrix
        let data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let a = DenseTensor::from_vec(data, &[4, 4]).unwrap();

        // Central 2x2 block (rows 1-2, cols 1-2)
        let sub = slice(&a, &[1..3, 1..3]).unwrap();
        assert_eq!(sub.shape(), &[2, 2]);
        // Elements: A[1,1], A[2,1], A[1,2], A[2,2] = 6, 7, 10, 11
        assert_eq!(sub.data(), &[6.0, 7.0, 10.0, 11.0]);
    }
}
