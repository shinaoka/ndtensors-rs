//! Stride computation utilities.
//!
//! Uses column-major (Fortran) order to match Julia's NDTensors.jl and faer.

/// Compute column-major strides from shape.
///
/// For shape [d0, d1, d2, ...], returns strides [1, d0, d0*d1, ...].
///
/// # Examples
///
/// ```
/// use ndtensors::strides::compute_strides;
///
/// assert_eq!(compute_strides(&[3, 4, 5]), vec![1, 3, 12]);
/// assert_eq!(compute_strides(&[2, 3]), vec![1, 2]);
/// assert_eq!(compute_strides(&[5]), vec![1]);
/// assert_eq!(compute_strides(&[]), vec![]);
/// ```
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1;

    for &dim in shape.iter() {
        strides.push(stride);
        stride *= dim;
    }

    strides
}

/// Convert cartesian indices to linear index using column-major order.
///
/// # Arguments
///
/// * `indices` - Cartesian indices for each dimension
/// * `strides` - Strides for each dimension
///
/// # Returns
///
/// Linear index into the underlying storage.
#[inline]
pub fn cartesian_to_linear(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides.iter())
        .map(|(&idx, &stride)| idx * stride)
        .sum()
}

/// Convert linear index to cartesian indices using column-major order.
///
/// # Arguments
///
/// * `linear` - Linear index
/// * `shape` - Shape of the tensor
///
/// # Returns
///
/// Cartesian indices for each dimension.
pub fn linear_to_cartesian(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = Vec::with_capacity(shape.len());

    for &dim in shape.iter() {
        indices.push(linear % dim);
        linear /= dim;
    }

    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_strides_3d() {
        assert_eq!(compute_strides(&[3, 4, 5]), vec![1, 3, 12]);
    }

    #[test]
    fn test_compute_strides_2d() {
        assert_eq!(compute_strides(&[2, 3]), vec![1, 2]);
    }

    #[test]
    fn test_compute_strides_1d() {
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_compute_strides_empty() {
        assert_eq!(compute_strides(&[]), vec![]);
    }

    #[test]
    fn test_cartesian_to_linear() {
        let strides = compute_strides(&[3, 4, 5]);
        // For shape [3, 4, 5] with column-major:
        // index [i, j, k] -> i + 3*j + 12*k
        assert_eq!(cartesian_to_linear(&[0, 0, 0], &strides), 0);
        assert_eq!(cartesian_to_linear(&[1, 0, 0], &strides), 1);
        assert_eq!(cartesian_to_linear(&[0, 1, 0], &strides), 3);
        assert_eq!(cartesian_to_linear(&[0, 0, 1], &strides), 12);
        assert_eq!(
            cartesian_to_linear(&[2, 3, 4], &strides),
            2 + 3 * 3 + 4 * 12
        );
    }

    #[test]
    fn test_linear_to_cartesian() {
        let shape = [3, 4, 5];
        assert_eq!(linear_to_cartesian(0, &shape), vec![0, 0, 0]);
        assert_eq!(linear_to_cartesian(1, &shape), vec![1, 0, 0]);
        assert_eq!(linear_to_cartesian(3, &shape), vec![0, 1, 0]);
        assert_eq!(linear_to_cartesian(12, &shape), vec![0, 0, 1]);
    }

    #[test]
    fn test_roundtrip() {
        let shape = [3, 4, 5];
        let strides = compute_strides(&shape);
        let total: usize = shape.iter().product();

        for linear in 0..total {
            let cartesian = linear_to_cartesian(linear, &shape);
            let back = cartesian_to_linear(&cartesian, &strides);
            assert_eq!(linear, back);
        }
    }
}
