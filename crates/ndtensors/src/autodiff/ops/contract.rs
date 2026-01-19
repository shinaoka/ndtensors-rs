//! Tracked contraction operation with backward pass.

use crate::autodiff::graph::{GradFn, NodeId, with_graph_f64};
use crate::autodiff::saved_tensor::SavedTensor;
use crate::autodiff::tensor::TrackedTensor;
use crate::contract::{contract, contract_vjp};
use crate::error::TensorError;
use crate::tensor::DenseTensor;

/// Backward function for tensor contraction.
///
/// Stores the inputs and labels from the forward pass to compute
/// gradients during backward.
#[derive(Debug)]
pub struct ContractBackward {
    /// Saved input A.
    saved_a: SavedTensor<f64>,
    /// Saved input B.
    saved_b: SavedTensor<f64>,
    /// Labels for A.
    labels_a: Vec<i32>,
    /// Labels for B.
    labels_b: Vec<i32>,
    /// Input node IDs (may be None for inputs that don't require grad).
    input_a_id: Option<NodeId>,
    input_b_id: Option<NodeId>,
}

impl GradFn<f64> for ContractBackward {
    fn backward(&self, grad_output: &DenseTensor<f64>) -> Vec<(NodeId, DenseTensor<f64>)> {
        // Get saved tensors
        let a = self.saved_a.materialize();
        let b = self.saved_b.materialize();

        // Compute VJP using existing implementation
        let (grad_a, grad_b) = contract_vjp(&a, &self.labels_a, &b, &self.labels_b, grad_output)
            .expect("contract_vjp failed in backward");

        // Return gradients for inputs that require grad
        let mut result = Vec::new();
        if let Some(id) = self.input_a_id {
            result.push((id, grad_a));
        }
        if let Some(id) = self.input_b_id {
            result.push((id, grad_b));
        }
        result
    }

    fn inputs(&self) -> Vec<NodeId> {
        let mut result = Vec::new();
        if let Some(id) = self.input_a_id {
            result.push(id);
        }
        if let Some(id) = self.input_b_id {
            result.push(id);
        }
        result
    }
}

/// Tracked tensor contraction with gradient computation.
///
/// Wraps `contract()` to track the computation for backward pass.
///
/// # Arguments
///
/// * `a` - First input tensor
/// * `labels_a` - Labels for each dimension of `a` (negative = contracted)
/// * `b` - Second input tensor
/// * `labels_b` - Labels for each dimension of `b` (negative = contracted)
///
/// # Returns
///
/// A tracked tensor containing the contraction result.
///
/// # Example
///
/// ```ignore
/// use ndtensors::autodiff::{TrackedTensor, tracked_contract, backward, clear_graph};
/// use ndtensors::Tensor;
///
/// clear_graph();
///
/// // Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
/// let a = TrackedTensor::leaf(Tensor::ones(&[2, 3]));
/// let b = TrackedTensor::leaf(Tensor::ones(&[3, 4]));
/// let c = tracked_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
///
/// assert_eq!(c.shape(), &[2, 4]);
/// assert!(c.requires_grad());
/// ```
pub fn tracked_contract(
    a: &TrackedTensor<f64>,
    labels_a: &[i32],
    b: &TrackedTensor<f64>,
    labels_b: &[i32],
) -> Result<TrackedTensor<f64>, TensorError> {
    // Forward pass
    let result = contract(a.tensor(), labels_a, b.tensor(), labels_b)?;

    // If neither input requires grad, no need to track
    if !a.requires_grad() && !b.requires_grad() {
        return Ok(TrackedTensor::new(result));
    }

    // Create backward function
    let backward = ContractBackward {
        saved_a: SavedTensor::new(Box::new(a.tensor().clone())),
        saved_b: SavedTensor::new(Box::new(b.tensor().clone())),
        labels_a: labels_a.to_vec(),
        labels_b: labels_b.to_vec(),
        input_a_id: if a.requires_grad() { a.node_id() } else { None },
        input_b_id: if b.requires_grad() { b.node_id() } else { None },
    };

    // Register in graph
    let requires_grad = a.requires_grad() || b.requires_grad();
    let node = with_graph_f64(|g| g.create_node(Box::new(backward), requires_grad));

    Ok(TrackedTensor::from_tensor_with_grad(result, node))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::autodiff::backward::backward;
    use crate::autodiff::tensor::clear_graph;
    use approx::assert_relative_eq;

    #[test]
    fn test_tracked_contract_no_grad() {
        let a = TrackedTensor::new(Tensor::ones(&[2, 3]));
        let b = TrackedTensor::new(Tensor::ones(&[3, 4]));

        let c = tracked_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert!(!c.requires_grad());
        assert_eq!(c.shape(), &[2, 4]);
    }

    #[test]
    fn test_tracked_contract_with_grad() {
        clear_graph();

        let a = TrackedTensor::leaf(Tensor::ones(&[2, 3]));
        let b = TrackedTensor::leaf(Tensor::ones(&[3, 4]));

        let c = tracked_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert!(c.requires_grad());
        assert_eq!(c.shape(), &[2, 4]);
    }

    #[test]
    fn test_tracked_contract_mixed_grad() {
        clear_graph();

        let a = TrackedTensor::leaf(Tensor::ones(&[2, 3]));
        let b = TrackedTensor::new(Tensor::ones(&[3, 4])); // No grad

        let c = tracked_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert!(c.requires_grad());
    }

    #[test]
    fn test_backward_matrix_multiply() {
        clear_graph();

        // A @ B where A is 2x3, B is 3x4
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 column-major
        let b_data: Vec<f64> = (1..=12).map(|x| x as f64).collect(); // 3x4

        let a = TrackedTensor::leaf(Tensor::from_vec(a_data, &[2, 3]).unwrap());
        let b = TrackedTensor::leaf(Tensor::from_vec(b_data, &[3, 4]).unwrap());

        // C = A @ B
        let c = tracked_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        // Sum to scalar: loss = sum(C)
        let ones = TrackedTensor::new(Tensor::ones(&[2, 4]));
        let loss = tracked_contract(&c, &[-1, -2], &ones, &[-1, -2]).unwrap();

        assert_eq!(loss.len(), 1);

        let grads = backward(&loss).unwrap();

        // Check gradients exist
        let grad_a = grads.get(a.node_id().unwrap()).unwrap();
        let grad_b = grads.get(b.node_id().unwrap()).unwrap();

        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_b.shape(), &[3, 4]);

        // grad_A = ones @ B^T = sum of rows of B for each column
        // For a 3x4 B matrix, B^T is 4x3, and ones@B^T gives row sums of B
        // Each element of grad_A should be sum of corresponding row of B
        // B = [[1,4,7,10], [2,5,8,11], [3,6,9,12]] in row-major
        // But stored column-major: [1,2,3,4,5,6,7,8,9,10,11,12]
        // Row sums: row0=[1,4,7,10]=22, row1=[2,5,8,11]=26, row2=[3,6,9,12]=30

        // grad_A[i,j] = sum_k B[j,k]
        // In column-major: grad_A[0,0] should be sum of row 0 of B = 22
        assert_relative_eq!(*grad_a.get(&[0, 0]).unwrap(), 22.0, epsilon = 1e-10);
        assert_relative_eq!(*grad_a.get(&[1, 0]).unwrap(), 22.0, epsilon = 1e-10);
        assert_relative_eq!(*grad_a.get(&[0, 1]).unwrap(), 26.0, epsilon = 1e-10);
        assert_relative_eq!(*grad_a.get(&[0, 2]).unwrap(), 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_backward_inner_product() {
        clear_graph();

        let a = TrackedTensor::leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap());
        let b = TrackedTensor::leaf(Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap());

        // loss = a . b (inner product = scalar)
        let loss = tracked_contract(&a, &[-1], &b, &[-1]).unwrap();

        assert_eq!(loss.len(), 1);
        // Inner product result: 1*4 + 2*5 + 3*6 = 32
        assert_relative_eq!(*loss.tensor().get_linear(0).unwrap(), 32.0, epsilon = 1e-10);

        let grads = backward(&loss).unwrap();

        // grad_a = b, grad_b = a
        let grad_a = grads.get(a.node_id().unwrap()).unwrap();
        let grad_b = grads.get(b.node_id().unwrap()).unwrap();

        assert_eq!(grad_a.data(), &[4.0, 5.0, 6.0]);
        assert_eq!(grad_b.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_backward_outer_product() {
        clear_graph();

        let a = TrackedTensor::leaf(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap());
        let b = TrackedTensor::leaf(Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap());

        // C = outer(a, b) with shape [2, 3]
        let c = tracked_contract(&a, &[1], &b, &[2]).unwrap();

        assert_eq!(c.shape(), &[2, 3]);

        // Sum to scalar
        let ones = TrackedTensor::new(Tensor::ones(&[2, 3]));
        let loss = tracked_contract(&c, &[-1, -2], &ones, &[-1, -2]).unwrap();

        let grads = backward(&loss).unwrap();

        // grad_a[i] = sum_j b[j] = sum(b) = 12
        // grad_b[j] = sum_i a[i] = sum(a) = 3
        let grad_a = grads.get(a.node_id().unwrap()).unwrap();
        let grad_b = grads.get(b.node_id().unwrap()).unwrap();

        assert_eq!(grad_a.data(), &[12.0, 12.0]);
        assert_eq!(grad_b.data(), &[3.0, 3.0, 3.0]);
    }
}
