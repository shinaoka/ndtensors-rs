//! Permutation backend trait.

use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Backend trait for DenseTensor permutation operations.
///
/// Implementations can provide optimized permutation algorithms:
/// - `GenericBackend`: Naive loop-based (always available)
/// - Future: `HpttBackend`: High-Performance Tensor Transpose library
/// - Future: `RayonBackend`: Parallel using rayon
///
/// This trait is specialized for DenseTensor, following NDTensors.jl's
/// storage-specific dispatch pattern.
pub trait PermutationBackend {
    /// Permute DenseTensor dimensions in-place.
    ///
    /// # Arguments
    ///
    /// * `dest` - Output DenseTensor (must have permuted shape)
    /// * `src` - Input DenseTensor
    /// * `perm` - Permutation of dimensions. `perm[i]` gives the source dimension
    ///   for the i-th dimension of the result.
    ///
    /// # Panics
    ///
    /// Panics if shapes don't match the permutation.
    fn permute_into<ElT: Scalar>(
        dest: &mut DenseTensor<ElT>,
        src: &DenseTensor<ElT>,
        perm: &[usize],
    );
}
