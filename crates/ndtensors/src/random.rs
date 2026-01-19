//! Random tensor construction.
//!
//! This module provides functions for creating tensors with random values.

use rand::Rng;
use rand::distr::StandardUniform;
use rand_distr::StandardNormal;

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
