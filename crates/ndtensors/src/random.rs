//! Random tensor construction.
//!
//! This module provides functions for creating tensors with random values,
//! including random orthogonal and unitary matrices.

use faer::linalg::solvers::Qr;
use rand::Rng;
use rand::distr::StandardUniform;
use rand_distr::StandardNormal;

use crate::backend::AsFaerMat;
use crate::scalar::{Scalar, c64};
use crate::storage::Dense;
use crate::tensor::Tensor;

/// Trait for types that can be randomly sampled from a uniform distribution.
pub trait RandomUniform: Scalar {
    /// Sample a random value from the uniform distribution [0, 1).
    fn sample_uniform<R: Rng>(rng: &mut R) -> Self;
}

impl RandomUniform for f64 {
    fn sample_uniform<R: Rng>(rng: &mut R) -> Self {
        rng.sample(StandardUniform)
    }
}

impl RandomUniform for c64 {
    fn sample_uniform<R: Rng>(rng: &mut R) -> Self {
        c64::new(rng.sample(StandardUniform), rng.sample(StandardUniform))
    }
}

/// Trait for types that can be randomly sampled from a normal distribution.
pub trait RandomNormal: Scalar {
    /// Sample a random value from the standard normal distribution.
    fn sample_normal<R: Rng>(rng: &mut R) -> Self;
}

impl RandomNormal for f64 {
    fn sample_normal<R: Rng>(rng: &mut R) -> Self {
        rng.sample(StandardNormal)
    }
}

impl RandomNormal for c64 {
    fn sample_normal<R: Rng>(rng: &mut R) -> Self {
        // Standard complex normal: real and imaginary parts are independent N(0, 1/2)
        // so that |z|^2 has mean 1
        let scale = std::f64::consts::FRAC_1_SQRT_2;
        c64::new(
            rng.sample::<f64, _>(StandardNormal) * scale,
            rng.sample::<f64, _>(StandardNormal) * scale,
        )
    }
}

impl<ElT: Scalar + RandomUniform> Tensor<ElT, Dense<ElT>> {
    /// Create a tensor with uniform random values in [0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::Tensor;
    ///
    /// let t: Tensor<f64> = Tensor::random(&[2, 3]);
    /// assert_eq!(t.shape(), &[2, 3]);
    ///
    /// // All values are in [0, 1)
    /// for i in 0..t.len() {
    ///     let v = *t.get_linear(i).unwrap();
    ///     assert!(v >= 0.0 && v < 1.0);
    /// }
    /// ```
    pub fn random(shape: &[usize]) -> Self {
        Self::random_with_rng(shape, &mut rand::rng())
    }

    /// Create a tensor with uniform random values using a specific RNG.
    ///
    /// This is useful for reproducible results with a seeded RNG.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::Tensor;
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let t1: Tensor<f64> = Tensor::random_with_rng(&[2, 3], &mut rng);
    ///
    /// // Reset RNG with same seed for reproducible results
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let t2: Tensor<f64> = Tensor::random_with_rng(&[2, 3], &mut rng);
    ///
    /// assert_eq!(t1.data(), t2.data());
    /// ```
    pub fn random_with_rng<R: Rng>(shape: &[usize], rng: &mut R) -> Self {
        let len: usize = shape.iter().product::<usize>().max(1);
        let data: Vec<ElT> = (0..len).map(|_| ElT::sample_uniform(rng)).collect();
        Self::from_vec(data, shape).expect("shape and data length should match")
    }
}

impl<ElT: Scalar + RandomNormal> Tensor<ElT, Dense<ElT>> {
    /// Create a tensor with standard normal random values.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::Tensor;
    ///
    /// let t: Tensor<f64> = Tensor::randn(&[2, 3]);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    pub fn randn(shape: &[usize]) -> Self {
        Self::randn_with_rng(shape, &mut rand::rng())
    }

    /// Create a tensor with standard normal random values using a specific RNG.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::Tensor;
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let t: Tensor<f64> = Tensor::randn_with_rng(&[2, 3], &mut rng);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    pub fn randn_with_rng<R: Rng>(shape: &[usize], rng: &mut R) -> Self {
        let len: usize = shape.iter().product::<usize>().max(1);
        let data: Vec<ElT> = (0..len).map(|_| ElT::sample_normal(rng)).collect();
        Self::from_vec(data, shape).expect("shape and data length should match")
    }
}

/// Generate a random orthogonal matrix of shape (n, m).
///
/// If n >= m, returns Q such that Q^T * Q = I_m (columns are orthonormal).
/// If n < m, returns Q such that Q * Q^T = I_n (rows are orthonormal).
///
/// Uses QR decomposition of a random normal matrix, following the approach
/// from NDTensors.jl based on <https://arxiv.org/abs/math-ph/0609050>.
/// The diagonal of R is made non-negative to ensure uniqueness.
///
/// # Example
///
/// ```
/// use ndtensors::random::random_orthog;
/// use approx::assert_relative_eq;
///
/// let (n, m) = (10, 4);
/// let o = random_orthog(n, m);
///
/// // Check O^T * O ≈ I_m
/// assert_eq!(o.shape(), &[n, m]);
/// for i in 0..m {
///     for j in 0..m {
///         let mut sum = 0.0;
///         for k in 0..n {
///             sum += o.get(&[k, i]).unwrap() * o.get(&[k, j]).unwrap();
///         }
///         let expected = if i == j { 1.0 } else { 0.0 };
///         assert_relative_eq!(sum, expected, epsilon = 1e-10);
///     }
/// }
/// ```
pub fn random_orthog(n: usize, m: usize) -> Tensor<f64, Dense<f64>> {
    random_orthog_with_rng(n, m, &mut rand::rng())
}

/// Generate a random orthogonal matrix with a specific RNG.
///
/// See [`random_orthog`] for details.
pub fn random_orthog_with_rng<R: Rng>(n: usize, m: usize, rng: &mut R) -> Tensor<f64, Dense<f64>> {
    if n < m {
        // Return transpose of random_orthog(m, n)
        let q = random_orthog_with_rng(m, n, rng);
        // Transpose: swap dimensions and reorder data
        let mut transposed = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                // Original is column-major: element (i, j) at index i + j*m
                // Transposed shape is (n, m), element (j, i) at index j + i*n
                transposed[j + i * n] = *q.get(&[i, j]).unwrap();
            }
        }
        return Tensor::from_vec(transposed, &[n, m]).unwrap();
    }

    // Generate random normal matrix
    let randn: Tensor<f64, Dense<f64>> = Tensor::randn_with_rng(&[n, m], rng);

    // Compute QR decomposition
    let mat = randn.as_faer_mat(n, m);
    let qr_result: Qr<f64> = Qr::new(mat);

    let k = n.min(m);

    // Get thin Q (n x k)
    let q_mat = qr_result.compute_thin_Q();

    // Get thin R to extract signs
    let r_mat = qr_result.thin_R();

    // Extract Q with sign correction to make R's diagonal non-negative
    let mut q_data = Vec::with_capacity(n * k);
    for j in 0..k {
        let sign = r_mat[(j, j)].signum();
        let sign = if sign == 0.0 { 1.0 } else { sign };
        for i in 0..n {
            q_data.push(q_mat[(i, j)] * sign);
        }
    }

    Tensor::from_vec(q_data, &[n, k]).unwrap()
}

/// Generate a random unitary matrix of shape (n, m).
///
/// If n >= m, returns U such that U^H * U = I_m (columns are orthonormal).
/// If n < m, returns U such that U * U^H = I_n (rows are orthonormal).
///
/// Uses QR decomposition of a random complex normal matrix, following the
/// approach from NDTensors.jl based on <https://arxiv.org/abs/math-ph/0609050>.
/// When n == m, the matrix is sampled according to the Haar measure.
///
/// # Example
///
/// ```
/// use ndtensors::random::random_unitary;
/// use ndtensors::c64;
/// use approx::assert_relative_eq;
///
/// let (n, m) = (10, 4);
/// let u = random_unitary(n, m);
///
/// // Check U^H * U ≈ I_m
/// assert_eq!(u.shape(), &[n, m]);
/// for i in 0..m {
///     for j in 0..m {
///         let mut sum = c64::new(0.0, 0.0);
///         for k in 0..n {
///             let u_ki = *u.get(&[k, i]).unwrap();
///             let u_kj = *u.get(&[k, j]).unwrap();
///             // U^H means conjugate transpose
///             sum = c64::new(
///                 sum.re + u_ki.re * u_kj.re + u_ki.im * u_kj.im,
///                 sum.im + u_ki.re * u_kj.im - u_ki.im * u_kj.re,
///             );
///         }
///         let expected_re = if i == j { 1.0 } else { 0.0 };
///         assert_relative_eq!(sum.re, expected_re, epsilon = 1e-10);
///         assert_relative_eq!(sum.im, 0.0, epsilon = 1e-10);
///     }
/// }
/// ```
pub fn random_unitary(n: usize, m: usize) -> Tensor<c64, Dense<c64>> {
    random_unitary_with_rng(n, m, &mut rand::rng())
}

/// Generate a random unitary matrix with a specific RNG.
///
/// See [`random_unitary`] for details.
pub fn random_unitary_with_rng<R: Rng>(n: usize, m: usize, rng: &mut R) -> Tensor<c64, Dense<c64>> {
    if n < m {
        // Return conjugate transpose of random_unitary(m, n)
        let u = random_unitary_with_rng(m, n, rng);
        // Conjugate transpose: swap dimensions and conjugate
        let mut transposed = vec![c64::new(0.0, 0.0); m * n];
        for i in 0..m {
            for j in 0..n {
                let val = *u.get(&[i, j]).unwrap();
                // Conjugate: (a + bi)^* = a - bi
                transposed[j + i * n] = c64::new(val.re, -val.im);
            }
        }
        return Tensor::from_vec(transposed, &[n, m]).unwrap();
    }

    // Generate random complex normal matrix
    let randn: Tensor<c64, Dense<c64>> = Tensor::randn_with_rng(&[n, m], rng);

    // Compute QR decomposition
    let mat = randn.as_faer_mat(n, m);
    let qr_result: Qr<c64> = Qr::new(mat);

    let k = n.min(m);

    // Get thin Q (n x k)
    let q_mat = qr_result.compute_thin_Q();

    // Get thin R to extract signs
    let r_mat = qr_result.thin_R();

    // Extract Q with sign correction to make R's diagonal real and non-negative
    let mut q_data = Vec::with_capacity(n * k);
    for j in 0..k {
        let r_jj = r_mat[(j, j)];
        let abs_r = (r_jj.re * r_jj.re + r_jj.im * r_jj.im).sqrt();
        // sign = r_jj / |r_jj| if |r_jj| > 0, else 1
        let sign = if abs_r > 1e-14 {
            c64::new(r_jj.re / abs_r, r_jj.im / abs_r)
        } else {
            c64::new(1.0, 0.0)
        };
        for i in 0..n {
            let q_ij = q_mat[(i, j)];
            // Multiply by sign: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
            q_data.push(c64::new(
                q_ij.re * sign.re - q_ij.im * sign.im,
                q_ij.re * sign.im + q_ij.im * sign.re,
            ));
        }
    }

    Tensor::from_vec(q_data, &[n, k]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_random_f64() {
        let t: Tensor<f64> = Tensor::random(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.len(), 6);

        // All values should be in [0, 1)
        for i in 0..t.len() {
            let v = *t.get_linear(i).unwrap();
            assert!((0.0..1.0).contains(&v), "value {} not in [0, 1)", v);
        }
    }

    #[test]
    fn test_random_c64() {
        let t: Tensor<c64> = Tensor::random(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.len(), 6);

        // Real and imaginary parts should be in [0, 1)
        for i in 0..t.len() {
            let v = t.get_linear(i).unwrap();
            assert!(
                (0.0..1.0).contains(&v.re),
                "real part {} not in [0, 1)",
                v.re
            );
            assert!(
                (0.0..1.0).contains(&v.im),
                "imaginary part {} not in [0, 1)",
                v.im
            );
        }
    }

    #[test]
    fn test_random_reproducible() {
        let mut rng1 = StdRng::seed_from_u64(12345);
        let t1: Tensor<f64> = Tensor::random_with_rng(&[3, 4], &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(12345);
        let t2: Tensor<f64> = Tensor::random_with_rng(&[3, 4], &mut rng2);

        assert_eq!(t1.data(), t2.data());
    }

    #[test]
    fn test_randn_f64() {
        let t: Tensor<f64> = Tensor::randn(&[100]);
        assert_eq!(t.shape(), &[100]);

        // Check that values are roughly normal (mean near 0, some variance)
        let sum: f64 = t.data().iter().sum();
        let mean = sum / 100.0;
        assert!(mean.abs() < 0.5, "mean {} too far from 0", mean);

        // Check variance is roughly 1
        let var: f64 = t.data().iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 100.0;
        assert!(var > 0.3 && var < 2.0, "variance {} too far from 1", var);
    }

    #[test]
    fn test_randn_c64() {
        let t: Tensor<c64> = Tensor::randn(&[100]);
        assert_eq!(t.shape(), &[100]);

        // For complex normal, |z|^2 should have mean 1
        let sum_sq: f64 = t.data().iter().map(|z| z.re * z.re + z.im * z.im).sum();
        let mean_sq = sum_sq / 100.0;
        assert!(
            mean_sq > 0.3 && mean_sq < 2.0,
            "mean |z|^2 {} too far from 1",
            mean_sq
        );
    }

    #[test]
    fn test_randn_reproducible() {
        let mut rng1 = StdRng::seed_from_u64(54321);
        let t1: Tensor<f64> = Tensor::randn_with_rng(&[3, 4], &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(54321);
        let t2: Tensor<f64> = Tensor::randn_with_rng(&[3, 4], &mut rng2);

        assert_eq!(t1.data(), t2.data());
    }

    #[test]
    fn test_random_scalar_tensor() {
        let t: Tensor<f64> = Tensor::random(&[]);
        assert_eq!(t.shape(), &[]);
        assert_eq!(t.len(), 1);
    }
}
