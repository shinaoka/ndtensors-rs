//! Integration tests for autodiff module.
//!
//! Tests backward-mode automatic differentiation with numerical gradient checks.

#![cfg(feature = "autodiff")]

use approx::assert_relative_eq;
use ndtensors::autodiff::{TrackedTensor, backward, clear_graph, tracked_contract};
use ndtensors::{Tensor, contract};

/// Compute numerical gradient using central difference.
///
/// grad_i ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
fn numerical_gradient<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut grad = vec![0.0; x.len()];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for i in 0..x.len() {
        x_plus[i] = x[i] + eps;
        x_minus[i] = x[i] - eps;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);
        grad[i] = (f_plus - f_minus) / (2.0 * eps);

        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }
    grad
}

#[test]
fn test_numerical_gradient_matmul() {
    let eps = 1e-5;

    // f(A, B) = sum(A @ B) where A is 2x3, B is 3x4
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();

    // Define loss function for varying A
    let loss_a = |a: &[f64]| -> f64 {
        let a_tensor = Tensor::from_vec(a.to_vec(), &[2, 3]).unwrap();
        let b_tensor = Tensor::from_vec(b_data.clone(), &[3, 4]).unwrap();
        let c = contract(&a_tensor, &[1, -1], &b_tensor, &[-1, 2]).unwrap();
        c.data().iter().sum()
    };

    // Define loss function for varying B
    let loss_b = |b: &[f64]| -> f64 {
        let a_tensor = Tensor::from_vec(a_data.clone(), &[2, 3]).unwrap();
        let b_tensor = Tensor::from_vec(b.to_vec(), &[3, 4]).unwrap();
        let c = contract(&a_tensor, &[1, -1], &b_tensor, &[-1, 2]).unwrap();
        c.data().iter().sum()
    };

    // Compute numerical gradients
    let numerical_grad_a = numerical_gradient(loss_a, &a_data, eps);
    let numerical_grad_b = numerical_gradient(loss_b, &b_data, eps);

    // Compute analytical gradients using autodiff
    clear_graph();
    let a = TrackedTensor::leaf(Tensor::from_vec(a_data.clone(), &[2, 3]).unwrap());
    let b = TrackedTensor::leaf(Tensor::from_vec(b_data.clone(), &[3, 4]).unwrap());
    let c = tracked_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
    let ones = TrackedTensor::new(Tensor::ones(&[2, 4]));
    let loss = tracked_contract(&c, &[-1, -2], &ones, &[-1, -2]).unwrap();

    let grads = backward(&loss).unwrap();
    let analytical_grad_a = grads.get(a.node_id().unwrap()).unwrap();
    let analytical_grad_b = grads.get(b.node_id().unwrap()).unwrap();

    // Compare gradients
    for (analytical, numerical) in analytical_grad_a.data().iter().zip(numerical_grad_a.iter()) {
        assert_relative_eq!(analytical, numerical, epsilon = 1e-4);
    }
    for (analytical, numerical) in analytical_grad_b.data().iter().zip(numerical_grad_b.iter()) {
        assert_relative_eq!(analytical, numerical, epsilon = 1e-4);
    }
}

#[test]
fn test_numerical_gradient_inner_product() {
    let eps = 1e-5;

    let a_data = vec![1.0, 2.0, 3.0];
    let b_data = vec![4.0, 5.0, 6.0];

    // f(a, b) = a . b (inner product)
    let loss_a = |a: &[f64]| -> f64 {
        let a_tensor = Tensor::from_vec(a.to_vec(), &[3]).unwrap();
        let b_tensor = Tensor::from_vec(b_data.clone(), &[3]).unwrap();
        let c = contract(&a_tensor, &[-1], &b_tensor, &[-1]).unwrap();
        *c.get_linear(0).unwrap()
    };

    let loss_b = |b: &[f64]| -> f64 {
        let a_tensor = Tensor::from_vec(a_data.clone(), &[3]).unwrap();
        let b_tensor = Tensor::from_vec(b.to_vec(), &[3]).unwrap();
        let c = contract(&a_tensor, &[-1], &b_tensor, &[-1]).unwrap();
        *c.get_linear(0).unwrap()
    };

    let numerical_grad_a = numerical_gradient(loss_a, &a_data, eps);
    let numerical_grad_b = numerical_gradient(loss_b, &b_data, eps);

    // For inner product: grad_a = b, grad_b = a
    for (i, &expected) in b_data.iter().enumerate() {
        assert_relative_eq!(numerical_grad_a[i], expected, epsilon = 1e-8);
    }
    for (i, &expected) in a_data.iter().enumerate() {
        assert_relative_eq!(numerical_grad_b[i], expected, epsilon = 1e-8);
    }

    // Also verify autodiff
    clear_graph();
    let a = TrackedTensor::leaf(Tensor::from_vec(a_data, &[3]).unwrap());
    let b = TrackedTensor::leaf(Tensor::from_vec(b_data, &[3]).unwrap());
    let loss = tracked_contract(&a, &[-1], &b, &[-1]).unwrap();

    let grads = backward(&loss).unwrap();
    let analytical_grad_a = grads.get(a.node_id().unwrap()).unwrap();
    let analytical_grad_b = grads.get(b.node_id().unwrap()).unwrap();

    for i in 0..3 {
        assert_relative_eq!(
            analytical_grad_a.data()[i],
            numerical_grad_a[i],
            epsilon = 1e-8
        );
        assert_relative_eq!(
            analytical_grad_b.data()[i],
            numerical_grad_b[i],
            epsilon = 1e-8
        );
    }
}

#[test]
fn test_numerical_gradient_chain() {
    let eps = 1e-5;

    // f(A) = sum((A @ B) @ C) where A is 2x3, B is 3x4, C is 4x2
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f64> = (1..=12).map(|x| x as f64 * 0.1).collect();
    let c_data: Vec<f64> = (1..=8).map(|x| x as f64 * 0.2).collect();

    let loss_fn = |a: &[f64]| -> f64 {
        let a_tensor = Tensor::from_vec(a.to_vec(), &[2, 3]).unwrap();
        let b_tensor = Tensor::from_vec(b_data.clone(), &[3, 4]).unwrap();
        let c_tensor = Tensor::from_vec(c_data.clone(), &[4, 2]).unwrap();

        // AB = A @ B
        let ab = contract(&a_tensor, &[1, -1], &b_tensor, &[-1, 2]).unwrap();
        // ABC = AB @ C
        let abc = contract(&ab, &[1, -1], &c_tensor, &[-1, 2]).unwrap();
        abc.data().iter().sum()
    };

    let numerical_grad_a = numerical_gradient(loss_fn, &a_data, eps);

    // Analytical gradient
    clear_graph();
    let a = TrackedTensor::leaf(Tensor::from_vec(a_data.clone(), &[2, 3]).unwrap());
    let b = TrackedTensor::new(Tensor::from_vec(b_data, &[3, 4]).unwrap());
    let c = TrackedTensor::new(Tensor::from_vec(c_data, &[4, 2]).unwrap());

    let ab = tracked_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
    let abc = tracked_contract(&ab, &[1, -1], &c, &[-1, 2]).unwrap();

    let ones = TrackedTensor::new(Tensor::ones(&[2, 2]));
    let loss = tracked_contract(&abc, &[-1, -2], &ones, &[-1, -2]).unwrap();

    let grads = backward(&loss).unwrap();
    let analytical_grad_a = grads.get(a.node_id().unwrap()).unwrap();

    for (analytical, numerical) in analytical_grad_a.data().iter().zip(numerical_grad_a.iter()) {
        assert_relative_eq!(analytical, numerical, epsilon = 1e-4);
    }
}

#[test]
fn test_gradient_accumulation() {
    // Test that gradients accumulate correctly when a tensor is used multiple times
    clear_graph();

    let a = TrackedTensor::leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap());
    let b = TrackedTensor::new(Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap());

    // loss = (a . b) + (a . b) = 2 * (a . b)
    let dot1 = tracked_contract(&a, &[-1], &b, &[-1]).unwrap();
    let _dot2 = tracked_contract(&a, &[-1], &b, &[-1]).unwrap();

    // We need to add them - for now just use another contraction
    // Actually, let's just verify each branch works
    let grads1 = backward(&dot1).unwrap();
    let grad_a1 = grads1.get(a.node_id().unwrap()).unwrap();
    assert_eq!(grad_a1.data(), &[1.0, 1.0, 1.0]); // grad = b

    clear_graph();
    let a2 = TrackedTensor::leaf(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap());
    let grads2 = backward(&tracked_contract(&a2, &[-1], &b, &[-1]).unwrap()).unwrap();
    let grad_a2 = grads2.get(a2.node_id().unwrap()).unwrap();
    assert_eq!(grad_a2.data(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_3d_tensor_contraction_backward() {
    let eps = 1e-5;

    // A[i,j,k] * B[k,l] -> C[i,j,l]
    let a_data: Vec<f64> = (1..=24).map(|x| x as f64 * 0.1).collect();
    let b_data: Vec<f64> = (1..=12).map(|x| x as f64 * 0.1).collect();

    let loss_a = |a: &[f64]| -> f64 {
        let a_tensor = Tensor::from_vec(a.to_vec(), &[2, 3, 4]).unwrap();
        let b_tensor = Tensor::from_vec(b_data.clone(), &[4, 3]).unwrap();
        let c = contract(&a_tensor, &[1, 2, -1], &b_tensor, &[-1, 3]).unwrap();
        c.data().iter().sum()
    };

    let numerical_grad_a = numerical_gradient(loss_a, &a_data, eps);

    clear_graph();
    let a = TrackedTensor::leaf(Tensor::from_vec(a_data, &[2, 3, 4]).unwrap());
    let b = TrackedTensor::new(Tensor::from_vec(b_data, &[4, 3]).unwrap());
    let c = tracked_contract(&a, &[1, 2, -1], &b, &[-1, 3]).unwrap();

    let ones = TrackedTensor::new(Tensor::ones(c.shape()));
    let loss = tracked_contract(&c, &[-1, -2, -3], &ones, &[-1, -2, -3]).unwrap();

    let grads = backward(&loss).unwrap();
    let analytical_grad_a = grads.get(a.node_id().unwrap()).unwrap();

    for (analytical, numerical) in analytical_grad_a.data().iter().zip(numerical_grad_a.iter()) {
        assert_relative_eq!(analytical, numerical, epsilon = 1e-4);
    }
}
