//! Saved tensor for backward pass.

use super::any_storage::AnyStorage;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;
use std::rc::Rc;

/// Policy for saving tensors during forward pass.
///
/// This enum controls how tensors are saved for backward computation.
/// For the PoC, only `SameDevice` is implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SavePolicy {
    /// Keep tensor in same device memory (default).
    ///
    /// This is the simplest policy and matches ITensors.jl's behavior.
    #[default]
    SameDevice,
    // Future: CpuOffload - move GPU tensors to CPU for memory efficiency
    // Future: Recompute - recompute during backward (activation checkpointing)
}

/// Saved tensor for backward pass.
///
/// Uses `Rc` for cheap cloning within the single-threaded computation graph.
/// Since the computation graph is thread-local, we don't need `Arc`.
#[derive(Debug)]
pub struct SavedTensor<T: Scalar> {
    /// The saved tensor data.
    data: Rc<Box<dyn AnyStorage<T>>>,
    /// Save policy used.
    policy: SavePolicy,
}

impl<T: Scalar> SavedTensor<T> {
    /// Create a new saved tensor with default policy (SameDevice).
    pub fn new(tensor: Box<dyn AnyStorage<T>>) -> Self {
        Self {
            data: Rc::new(tensor),
            policy: SavePolicy::default(),
        }
    }

    /// Create with explicit save policy.
    pub fn with_policy(tensor: Box<dyn AnyStorage<T>>, policy: SavePolicy) -> Self {
        Self {
            data: Rc::new(tensor),
            policy,
        }
    }

    /// Get reference to saved data.
    pub fn get(&self) -> &dyn AnyStorage<T> {
        self.data.as_ref().as_ref()
    }

    /// Materialize saved tensor as DenseTensor.
    ///
    /// For SameDevice policy, this is just a conversion.
    /// Future policies (CpuOffload, Recompute) would have different behaviors.
    pub fn materialize(&self) -> DenseTensor<T> {
        self.data.to_dense_cpu()
    }

    /// Get save policy.
    pub fn policy(&self) -> SavePolicy {
        self.policy
    }
}

impl<T: Scalar> Clone for SavedTensor<T> {
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
            policy: self.policy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_saved_tensor_new() {
        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let saved = SavedTensor::new(Box::new(t));

        assert_eq!(saved.policy(), SavePolicy::SameDevice);
        assert_eq!(saved.get().len(), 3);
    }

    #[test]
    fn test_saved_tensor_clone_shares_data() {
        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let saved1 = SavedTensor::new(Box::new(t));
        let saved2 = saved1.clone();

        // Both should point to the same Rc
        assert!(Rc::ptr_eq(&saved1.data, &saved2.data));
    }

    #[test]
    fn test_saved_tensor_materialize() {
        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let saved = SavedTensor::new(Box::new(t.clone()));

        let materialized = saved.materialize();
        assert_eq!(materialized.data(), t.data());
    }

    #[test]
    fn test_save_policy_default() {
        assert_eq!(SavePolicy::default(), SavePolicy::SameDevice);
    }
}
