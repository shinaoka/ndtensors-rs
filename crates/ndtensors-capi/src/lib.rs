//! C API for ndtensors
//!
//! This crate provides a C-compatible interface to the ndtensors library,
//! allowing it to be called from Julia, Python, C, and other languages.
//!
//! All extern "C" functions are inherently unsafe as they work with raw pointers
//! from foreign code. The `#[unsafe(no_mangle)]` attribute marks the entire
//! function signature as unsafe at the FFI boundary.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use libc::{c_double, c_int, c_long, size_t};
use ndtensors::{Tensor, contract, contract_jvp, contract_vjp};
use std::panic::catch_unwind;
use std::ptr;

// Status codes
pub type StatusCode = c_int;

pub const NDT_SUCCESS: StatusCode = 0;
pub const NDT_INVALID_ARGUMENT: StatusCode = -1;
pub const NDT_SHAPE_MISMATCH: StatusCode = -2;
pub const NDT_INDEX_OUT_OF_BOUNDS: StatusCode = -3;
pub const NDT_INTERNAL_ERROR: StatusCode = -4;
pub const NDT_INVALID_PERMUTATION: StatusCode = -5;

/// Opaque tensor type for f64
#[repr(C)]
pub struct ndt_tensor_f64 {
    _private: *mut std::ffi::c_void,
}

impl ndt_tensor_f64 {
    fn from_tensor(tensor: Tensor<f64>) -> Self {
        let boxed = Box::new(tensor);
        Self {
            _private: Box::into_raw(boxed) as *mut std::ffi::c_void,
        }
    }

    fn inner(&self) -> &Tensor<f64> {
        unsafe { &*(self._private as *const Tensor<f64>) }
    }

    fn inner_mut(&mut self) -> &mut Tensor<f64> {
        unsafe { &mut *(self._private as *mut Tensor<f64>) }
    }
}

impl Drop for ndt_tensor_f64 {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut Tensor<f64>);
            }
        }
    }
}

impl Clone for ndt_tensor_f64 {
    fn clone(&self) -> Self {
        let tensor = self.inner().clone();
        Self::from_tensor(tensor)
    }
}

// ============================================================================
// Tensor creation functions
// ============================================================================

/// Create a new tensor filled with zeros.
///
/// # Arguments
/// * `shape` - Pointer to array of dimensions
/// * `ndim` - Number of dimensions
/// * `status` - Pointer to receive status code
///
/// # Returns
/// Pointer to new tensor, or null on error
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_zeros(
    shape: *const size_t,
    ndim: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64 {
    if status.is_null() {
        return ptr::null_mut();
    }

    if shape.is_null() && ndim > 0 {
        unsafe {
            *status = NDT_INVALID_ARGUMENT;
        }
        return ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let shape_slice = if ndim == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(shape, ndim) }
        };

        let tensor = Tensor::<f64>::zeros(shape_slice);
        Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(tensor)))
    });

    match result {
        Ok(ptr) => {
            unsafe {
                *status = NDT_SUCCESS;
            }
            ptr
        }
        Err(_) => {
            unsafe {
                *status = NDT_INTERNAL_ERROR;
            }
            ptr::null_mut()
        }
    }
}

/// Create a new tensor filled with ones.
///
/// # Arguments
/// * `shape` - Pointer to array of dimensions
/// * `ndim` - Number of dimensions
/// * `status` - Pointer to receive status code
///
/// # Returns
/// Pointer to new tensor, or null on error
///
/// # Ownership
/// On success, the caller is responsible for releasing the tensor
/// by calling `ndt_tensor_f64_release`.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_ones(
    shape: *const size_t,
    ndim: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64 {
    if status.is_null() {
        return ptr::null_mut();
    }

    if shape.is_null() && ndim > 0 {
        unsafe {
            *status = NDT_INVALID_ARGUMENT;
        }
        return ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let shape_slice = if ndim == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(shape, ndim) }
        };

        let tensor = Tensor::<f64>::ones(shape_slice);
        Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(tensor)))
    });

    match result {
        Ok(ptr) => {
            unsafe {
                *status = NDT_SUCCESS;
            }
            ptr
        }
        Err(_) => {
            unsafe {
                *status = NDT_INTERNAL_ERROR;
            }
            ptr::null_mut()
        }
    }
}

/// Create a new tensor with uniform random values in [0, 1).
///
/// # Arguments
/// * `shape` - Pointer to array of dimensions
/// * `ndim` - Number of dimensions
/// * `status` - Pointer to receive status code
///
/// # Returns
/// Pointer to new tensor, or null on error
///
/// # Ownership
/// On success, the caller is responsible for releasing the tensor
/// by calling `ndt_tensor_f64_release`.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_rand(
    shape: *const size_t,
    ndim: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64 {
    if status.is_null() {
        return ptr::null_mut();
    }

    if shape.is_null() && ndim > 0 {
        unsafe {
            *status = NDT_INVALID_ARGUMENT;
        }
        return ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let shape_slice = if ndim == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(shape, ndim) }
        };

        let tensor = Tensor::<f64>::random(shape_slice);
        Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(tensor)))
    });

    match result {
        Ok(ptr) => {
            unsafe {
                *status = NDT_SUCCESS;
            }
            ptr
        }
        Err(_) => {
            unsafe {
                *status = NDT_INTERNAL_ERROR;
            }
            ptr::null_mut()
        }
    }
}

/// Create a new tensor with standard normal random values (mean=0, std=1).
///
/// # Arguments
/// * `shape` - Pointer to array of dimensions
/// * `ndim` - Number of dimensions
/// * `status` - Pointer to receive status code
///
/// # Returns
/// Pointer to new tensor, or null on error
///
/// # Ownership
/// On success, the caller is responsible for releasing the tensor
/// by calling `ndt_tensor_f64_release`.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_randn(
    shape: *const size_t,
    ndim: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64 {
    if status.is_null() {
        return ptr::null_mut();
    }

    if shape.is_null() && ndim > 0 {
        unsafe {
            *status = NDT_INVALID_ARGUMENT;
        }
        return ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let shape_slice = if ndim == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(shape, ndim) }
        };

        let tensor = Tensor::<f64>::randn(shape_slice);
        Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(tensor)))
    });

    match result {
        Ok(ptr) => {
            unsafe {
                *status = NDT_SUCCESS;
            }
            ptr
        }
        Err(_) => {
            unsafe {
                *status = NDT_INTERNAL_ERROR;
            }
            ptr::null_mut()
        }
    }
}

/// Create a new tensor from data.
///
/// # Arguments
/// * `data` - Pointer to data array (column-major order)
/// * `len` - Length of data array
/// * `shape` - Pointer to array of dimensions
/// * `ndim` - Number of dimensions
/// * `status` - Pointer to receive status code
///
/// # Returns
/// Pointer to new tensor, or null on error
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_from_data(
    data: *const c_double,
    len: size_t,
    shape: *const size_t,
    ndim: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64 {
    if status.is_null() {
        return ptr::null_mut();
    }

    if data.is_null() || (shape.is_null() && ndim > 0) {
        unsafe {
            *status = NDT_INVALID_ARGUMENT;
        }
        return ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
        let shape_slice = if ndim == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(shape, ndim) }
        };

        match Tensor::<f64>::from_vec(data_slice.to_vec(), shape_slice) {
            Ok(tensor) => {
                let ptr = Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(tensor)));
                (ptr, NDT_SUCCESS)
            }
            Err(_) => (ptr::null_mut(), NDT_SHAPE_MISMATCH),
        }
    });

    match result {
        Ok((ptr, code)) => {
            unsafe {
                *status = code;
            }
            ptr
        }
        Err(_) => {
            unsafe {
                *status = NDT_INTERNAL_ERROR;
            }
            ptr::null_mut()
        }
    }
}

// ============================================================================
// Tensor lifecycle functions
// ============================================================================

/// Release (free) a tensor.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_release(tensor: *mut ndt_tensor_f64) {
    if !tensor.is_null() {
        unsafe {
            let _ = Box::from_raw(tensor);
        }
    }
}

/// Clone a tensor.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_clone(src: *const ndt_tensor_f64) -> *mut ndt_tensor_f64 {
    if src.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let src_ref = &*src;
        let cloned = src_ref.clone();
        Box::into_raw(Box::new(cloned))
    }));

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// Tensor query functions
// ============================================================================

/// Get the number of dimensions.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_ndim(tensor: *const ndt_tensor_f64) -> size_t {
    if tensor.is_null() {
        return 0;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        (*tensor).inner().ndim()
    }));

    result.unwrap_or(0)
}

/// Get the total number of elements.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_len(tensor: *const ndt_tensor_f64) -> size_t {
    if tensor.is_null() {
        return 0;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        (*tensor).inner().len()
    }));

    result.unwrap_or(0)
}

/// Get the shape (dimensions).
///
/// # Arguments
/// * `tensor` - Tensor pointer
/// * `out` - Output array for shape (must have space for ndim elements)
///
/// # Returns
/// Status code
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_shape(
    tensor: *const ndt_tensor_f64,
    out: *mut size_t,
) -> StatusCode {
    if tensor.is_null() || out.is_null() {
        return NDT_INVALID_ARGUMENT;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let t = (*tensor).inner();
        let shape = t.shape();
        for (i, &dim) in shape.iter().enumerate() {
            *out.add(i) = dim;
        }
        NDT_SUCCESS
    }));

    result.unwrap_or(NDT_INTERNAL_ERROR)
}

/// Get pointer to underlying data (read-only).
///
/// # Safety
/// The returned pointer is only valid while the tensor exists.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_data(tensor: *const ndt_tensor_f64) -> *const c_double {
    if tensor.is_null() {
        return ptr::null();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        (*tensor).inner().data().as_ptr()
    }));

    result.unwrap_or(ptr::null())
}

/// Get pointer to underlying data (mutable).
///
/// # Safety
/// The returned pointer is only valid while the tensor exists.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_data_mut(tensor: *mut ndt_tensor_f64) -> *mut c_double {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        (*tensor).inner_mut().data_mut().as_mut_ptr()
    }));

    result.unwrap_or(ptr::null_mut())
}

// ============================================================================
// Element access functions
// ============================================================================

/// Get element by linear index.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_get_linear(
    tensor: *const ndt_tensor_f64,
    index: size_t,
    out: *mut c_double,
) -> StatusCode {
    if tensor.is_null() || out.is_null() {
        return NDT_INVALID_ARGUMENT;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let t = (*tensor).inner();
        match t.get_linear(index) {
            Some(&val) => {
                *out = val;
                NDT_SUCCESS
            }
            None => NDT_INDEX_OUT_OF_BOUNDS,
        }
    }));

    result.unwrap_or(NDT_INTERNAL_ERROR)
}

/// Set element by linear index.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_set_linear(
    tensor: *mut ndt_tensor_f64,
    index: size_t,
    value: c_double,
) -> StatusCode {
    if tensor.is_null() {
        return NDT_INVALID_ARGUMENT;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let t = (*tensor).inner_mut();
        match t.get_linear_mut(index) {
            Some(elem) => {
                *elem = value;
                NDT_SUCCESS
            }
            None => NDT_INDEX_OUT_OF_BOUNDS,
        }
    }));

    result.unwrap_or(NDT_INTERNAL_ERROR)
}

/// Fill tensor with a value.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_fill(tensor: *mut ndt_tensor_f64, value: c_double) -> StatusCode {
    if tensor.is_null() {
        return NDT_INVALID_ARGUMENT;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        (*tensor).inner_mut().fill(value);
        NDT_SUCCESS
    }));

    result.unwrap_or(NDT_INTERNAL_ERROR)
}

// ============================================================================
// Tensor operations
// ============================================================================

/// Permute tensor dimensions.
///
/// # Arguments
/// * `tensor` - Input tensor
/// * `perm` - Permutation array (perm[i] = source dimension for output dimension i)
/// * `ndim` - Number of dimensions (must match tensor's ndim)
/// * `status` - Pointer to receive status code
///
/// # Returns
/// New tensor with permuted dimensions, or null on error
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_permutedims(
    tensor: *const ndt_tensor_f64,
    perm: *const size_t,
    ndim: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64 {
    if status.is_null() {
        return ptr::null_mut();
    }

    if tensor.is_null() || perm.is_null() {
        unsafe {
            *status = NDT_INVALID_ARGUMENT;
        }
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let t = (*tensor).inner();
        let perm_slice = std::slice::from_raw_parts(perm, ndim);

        match t.permutedims(perm_slice) {
            Ok(permuted) => {
                let ptr = Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(permuted)));
                (ptr, NDT_SUCCESS)
            }
            Err(_) => (ptr::null_mut(), NDT_INVALID_PERMUTATION),
        }
    }));

    match result {
        Ok((ptr, code)) => {
            unsafe {
                *status = code;
            }
            ptr
        }
        Err(_) => {
            unsafe {
                *status = NDT_INTERNAL_ERROR;
            }
            ptr::null_mut()
        }
    }
}

/// Contract two tensors using label-based contraction.
///
/// # Arguments
/// * `a` - First tensor
/// * `labels_a` - Labels for each dimension of `a` (negative = contracted, positive = output)
/// * `ndim_a` - Number of dimensions of `a`
/// * `b` - Second tensor
/// * `labels_b` - Labels for each dimension of `b`
/// * `ndim_b` - Number of dimensions of `b`
/// * `status` - Pointer to receive status code
///
/// # Returns
/// New tensor with contracted result, or null on error
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_contract(
    a: *const ndt_tensor_f64,
    labels_a: *const c_long,
    ndim_a: size_t,
    b: *const ndt_tensor_f64,
    labels_b: *const c_long,
    ndim_b: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64 {
    if status.is_null() {
        return ptr::null_mut();
    }

    if a.is_null() || b.is_null() || labels_a.is_null() || labels_b.is_null() {
        unsafe {
            *status = NDT_INVALID_ARGUMENT;
        }
        return ptr::null_mut();
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let tensor_a = (*a).inner();
        let tensor_b = (*b).inner();
        let labels_a_slice = std::slice::from_raw_parts(labels_a, ndim_a);
        let labels_b_slice = std::slice::from_raw_parts(labels_b, ndim_b);

        // Convert c_long (i64 on most platforms) to i32
        let labels_a_i32: Vec<i32> = labels_a_slice.iter().map(|&l| l as i32).collect();
        let labels_b_i32: Vec<i32> = labels_b_slice.iter().map(|&l| l as i32).collect();

        match contract(tensor_a, &labels_a_i32, tensor_b, &labels_b_i32) {
            Ok(result) => {
                let ptr = Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(result)));
                (ptr, NDT_SUCCESS)
            }
            Err(_) => (ptr::null_mut(), NDT_SHAPE_MISMATCH),
        }
    }));

    match result {
        Ok((ptr, code)) => {
            unsafe {
                *status = code;
            }
            ptr
        }
        Err(_) => {
            unsafe {
                *status = NDT_INTERNAL_ERROR;
            }
            ptr::null_mut()
        }
    }
}

/// Compute VJP (Vector-Jacobian Product) for tensor contraction.
///
/// Given the forward pass `c = contract(a, labels_a, b, labels_b)` and the gradient
/// of the loss with respect to `c` (`grad_output`), this computes the gradients
/// with respect to `a` and `b`.
///
/// # Arguments
/// * `a` - First tensor from forward pass
/// * `labels_a` - Labels for each dimension of `a`
/// * `ndim_a` - Number of dimensions of `a`
/// * `b` - Second tensor from forward pass
/// * `labels_b` - Labels for each dimension of `b`
/// * `ndim_b` - Number of dimensions of `b`
/// * `grad_output` - Gradient of loss with respect to output
/// * `grad_a_out` - Output: pointer to receive grad_a tensor
/// * `grad_b_out` - Output: pointer to receive grad_b tensor
/// * `status` - Pointer to receive status code
///
/// # Ownership
/// On success, `grad_a_out` and `grad_b_out` will point to newly allocated tensors.
/// The caller is responsible for releasing these tensors by calling
/// `ndt_tensor_f64_release` when they are no longer needed.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_contract_vjp(
    a: *const ndt_tensor_f64,
    labels_a: *const c_long,
    ndim_a: size_t,
    b: *const ndt_tensor_f64,
    labels_b: *const c_long,
    ndim_b: size_t,
    grad_output: *const ndt_tensor_f64,
    grad_a_out: *mut *mut ndt_tensor_f64,
    grad_b_out: *mut *mut ndt_tensor_f64,
    status: *mut StatusCode,
) {
    if status.is_null() {
        return;
    }

    if a.is_null()
        || b.is_null()
        || grad_output.is_null()
        || labels_a.is_null()
        || labels_b.is_null()
        || grad_a_out.is_null()
        || grad_b_out.is_null()
    {
        unsafe {
            *status = NDT_INVALID_ARGUMENT;
        }
        return;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let tensor_a = (*a).inner();
        let tensor_b = (*b).inner();
        let tensor_grad_output = (*grad_output).inner();
        let labels_a_slice = std::slice::from_raw_parts(labels_a, ndim_a);
        let labels_b_slice = std::slice::from_raw_parts(labels_b, ndim_b);

        // Convert c_long to i32
        let labels_a_i32: Vec<i32> = labels_a_slice.iter().map(|&l| l as i32).collect();
        let labels_b_i32: Vec<i32> = labels_b_slice.iter().map(|&l| l as i32).collect();

        match contract_vjp(
            tensor_a,
            &labels_a_i32,
            tensor_b,
            &labels_b_i32,
            tensor_grad_output,
        ) {
            Ok((grad_a, grad_b)) => {
                let grad_a_ptr = Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(grad_a)));
                let grad_b_ptr = Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(grad_b)));
                *grad_a_out = grad_a_ptr;
                *grad_b_out = grad_b_ptr;
                NDT_SUCCESS
            }
            Err(_) => {
                *grad_a_out = ptr::null_mut();
                *grad_b_out = ptr::null_mut();
                NDT_SHAPE_MISMATCH
            }
        }
    }));

    match result {
        Ok(code) => unsafe {
            *status = code;
        },
        Err(_) => unsafe {
            *status = NDT_INTERNAL_ERROR;
            *grad_a_out = ptr::null_mut();
            *grad_b_out = ptr::null_mut();
        },
    }
}

/// Compute JVP (Jacobian-vector product) for tensor contraction.
///
/// Given the contraction `c = contract(a, b)`, computes both the primal
/// result and the tangent (JVP) using the Leibniz rule:
///   tangent_c = contract(tangent_a, b) + contract(a, tangent_b)
///
/// This is the forward-mode autodiff primitive for tensor contraction.
///
/// # Arguments
///
/// * `a` - First tensor (primal)
/// * `labels_a` - Labels for dimensions of `a`
/// * `ndim_a` - Number of dimensions of `a`
/// * `b` - Second tensor (primal)
/// * `labels_b` - Labels for dimensions of `b`
/// * `ndim_b` - Number of dimensions of `b`
/// * `tangent_a` - Tangent for `a` (NULL for zero tangent)
/// * `tangent_b` - Tangent for `b` (NULL for zero tangent)
/// * `primal_out` - Output: primal result (contract(a, b))
/// * `tangent_out` - Output: tangent result (will be NULL if both tangent inputs are NULL)
/// * `status` - Pointer to receive status code
///
/// # Ownership
///
/// On success, `primal_out` will point to a newly allocated tensor.
/// `tangent_out` will point to a newly allocated tensor, or be NULL
/// if both `tangent_a` and `tangent_b` are NULL.
/// The caller is responsible for releasing these tensors by calling
/// `ndt_tensor_f64_release` when they are no longer needed.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_contract_jvp(
    a: *const ndt_tensor_f64,
    labels_a: *const c_long,
    ndim_a: size_t,
    b: *const ndt_tensor_f64,
    labels_b: *const c_long,
    ndim_b: size_t,
    tangent_a: *const ndt_tensor_f64,
    tangent_b: *const ndt_tensor_f64,
    primal_out: *mut *mut ndt_tensor_f64,
    tangent_out: *mut *mut ndt_tensor_f64,
    status: *mut StatusCode,
) {
    if status.is_null() {
        return;
    }

    // Validate required arguments
    if a.is_null()
        || b.is_null()
        || labels_a.is_null()
        || labels_b.is_null()
        || primal_out.is_null()
        || tangent_out.is_null()
    {
        unsafe {
            *status = NDT_INVALID_ARGUMENT;
        }
        return;
    }

    let result = catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let tensor_a = (*a).inner();
        let tensor_b = (*b).inner();
        let labels_a_slice = std::slice::from_raw_parts(labels_a, ndim_a);
        let labels_b_slice = std::slice::from_raw_parts(labels_b, ndim_b);

        // Convert labels
        let labels_a_i32: Vec<i32> = labels_a_slice.iter().map(|&l| l as i32).collect();
        let labels_b_i32: Vec<i32> = labels_b_slice.iter().map(|&l| l as i32).collect();

        // Get optional tangents
        let tangent_a_opt = if tangent_a.is_null() {
            None
        } else {
            Some((*tangent_a).inner())
        };
        let tangent_b_opt = if tangent_b.is_null() {
            None
        } else {
            Some((*tangent_b).inner())
        };

        // Compute primal
        let primal = match contract(tensor_a, &labels_a_i32, tensor_b, &labels_b_i32) {
            Ok(p) => p,
            Err(_) => return NDT_SHAPE_MISMATCH,
        };

        // Compute JVP
        let tangent = match contract_jvp(
            tensor_a,
            &labels_a_i32,
            tensor_b,
            &labels_b_i32,
            tangent_a_opt,
            tangent_b_opt,
        ) {
            Ok(t) => t,
            Err(_) => return NDT_SHAPE_MISMATCH,
        };

        // Allocate outputs
        *primal_out = Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(primal)));
        *tangent_out = match tangent {
            Some(t) => Box::into_raw(Box::new(ndt_tensor_f64::from_tensor(t))),
            None => ptr::null_mut(),
        };

        NDT_SUCCESS
    }));

    match result {
        Ok(code) => unsafe {
            *status = code;
        },
        Err(_) => unsafe {
            *status = NDT_INTERNAL_ERROR;
            *primal_out = ptr::null_mut();
            *tangent_out = ptr::null_mut();
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let shape = [2usize, 3usize];
        let mut status: StatusCode = -999;

        let tensor = ndt_tensor_f64_zeros(shape.as_ptr(), 2, &mut status);
        assert_eq!(status, NDT_SUCCESS);
        assert!(!tensor.is_null());

        assert_eq!(ndt_tensor_f64_ndim(tensor), 2);
        assert_eq!(ndt_tensor_f64_len(tensor), 6);

        let mut out_shape = [0usize; 2];
        assert_eq!(
            ndt_tensor_f64_shape(tensor, out_shape.as_mut_ptr()),
            NDT_SUCCESS
        );
        assert_eq!(out_shape, [2, 3]);

        ndt_tensor_f64_release(tensor);
    }

    #[test]
    fn test_tensor_from_data() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f64];
        let shape = [2usize, 3usize];
        let mut status: StatusCode = -999;

        let tensor =
            ndt_tensor_f64_from_data(data.as_ptr(), data.len(), shape.as_ptr(), 2, &mut status);
        assert_eq!(status, NDT_SUCCESS);
        assert!(!tensor.is_null());

        let mut val = 0.0;
        assert_eq!(ndt_tensor_f64_get_linear(tensor, 0, &mut val), NDT_SUCCESS);
        assert_eq!(val, 1.0);

        assert_eq!(ndt_tensor_f64_get_linear(tensor, 5, &mut val), NDT_SUCCESS);
        assert_eq!(val, 6.0);

        ndt_tensor_f64_release(tensor);
    }

    #[test]
    fn test_tensor_set_linear() {
        let shape = [2usize, 3usize];
        let mut status: StatusCode = -999;

        let tensor = ndt_tensor_f64_zeros(shape.as_ptr(), 2, &mut status);
        assert_eq!(status, NDT_SUCCESS);

        assert_eq!(ndt_tensor_f64_set_linear(tensor, 3, 42.0), NDT_SUCCESS);

        let mut val = 0.0;
        assert_eq!(ndt_tensor_f64_get_linear(tensor, 3, &mut val), NDT_SUCCESS);
        assert_eq!(val, 42.0);

        ndt_tensor_f64_release(tensor);
    }

    #[test]
    fn test_tensor_clone() {
        let shape = [2usize, 3usize];
        let mut status: StatusCode = -999;

        let tensor = ndt_tensor_f64_zeros(shape.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(tensor, 5.0), NDT_SUCCESS);

        let cloned = ndt_tensor_f64_clone(tensor);
        assert!(!cloned.is_null());

        let mut val = 0.0;
        assert_eq!(ndt_tensor_f64_get_linear(cloned, 0, &mut val), NDT_SUCCESS);
        assert_eq!(val, 5.0);

        ndt_tensor_f64_release(tensor);
        ndt_tensor_f64_release(cloned);
    }

    #[test]
    fn test_tensor_permutedims() {
        // Create 2x3 tensor: [[1,3,5], [2,4,6]] in column-major
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f64];
        let shape = [2usize, 3usize];
        let mut status: StatusCode = -999;

        let tensor =
            ndt_tensor_f64_from_data(data.as_ptr(), data.len(), shape.as_ptr(), 2, &mut status);
        assert_eq!(status, NDT_SUCCESS);

        // Transpose: [0,1] -> [1,0]
        let perm = [1usize, 0usize];
        let transposed = ndt_tensor_f64_permutedims(tensor, perm.as_ptr(), 2, &mut status);
        assert_eq!(status, NDT_SUCCESS);
        assert!(!transposed.is_null());

        // Check new shape is 3x2
        assert_eq!(ndt_tensor_f64_ndim(transposed), 2);
        let mut new_shape = [0usize; 2];
        ndt_tensor_f64_shape(transposed, new_shape.as_mut_ptr());
        assert_eq!(new_shape, [3, 2]);

        // Original t[0,1] = 3.0 should be at transposed[1,0]
        // Linear index for [1,0] in 3x2 tensor = 1
        let mut val = 0.0;
        assert_eq!(
            ndt_tensor_f64_get_linear(transposed, 1, &mut val),
            NDT_SUCCESS
        );
        assert_eq!(val, 3.0);

        ndt_tensor_f64_release(tensor);
        ndt_tensor_f64_release(transposed);
    }

    #[test]
    fn test_tensor_contract() {
        // Matrix multiplication: A[2x3] * B[3x4] = C[2x4]
        let mut status: StatusCode = -999;

        // Create A: 2x3 matrix of ones
        let shape_a = [2usize, 3usize];
        let a = ndt_tensor_f64_zeros(shape_a.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(a, 1.0), NDT_SUCCESS);

        // Create B: 3x4 matrix of ones
        let shape_b = [3usize, 4usize];
        let b = ndt_tensor_f64_zeros(shape_b.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(b, 1.0), NDT_SUCCESS);

        // Contract: A[1,-1] * B[-1,2] = C[1,2]
        let labels_a = [1i64, -1i64];
        let labels_b = [-1i64, 2i64];

        let c = ndt_tensor_f64_contract(
            a,
            labels_a.as_ptr() as *const c_long,
            2,
            b,
            labels_b.as_ptr() as *const c_long,
            2,
            &mut status,
        );
        assert_eq!(status, NDT_SUCCESS);
        assert!(!c.is_null());

        // Check shape is 2x4
        assert_eq!(ndt_tensor_f64_ndim(c), 2);
        let mut out_shape = [0usize; 2];
        ndt_tensor_f64_shape(c, out_shape.as_mut_ptr());
        assert_eq!(out_shape, [2, 4]);

        // Each element should be 3 (sum over contracted dimension of size 3)
        let mut val = 0.0;
        assert_eq!(ndt_tensor_f64_get_linear(c, 0, &mut val), NDT_SUCCESS);
        assert_eq!(val, 3.0);

        ndt_tensor_f64_release(a);
        ndt_tensor_f64_release(b);
        ndt_tensor_f64_release(c);
    }

    #[test]
    fn test_tensor_contract_vjp() {
        // Matrix multiplication VJP: A[2x3] * B[3x4] = C[2x4]
        let mut status: StatusCode = -999;

        // Create A: 2x3 matrix of ones
        let shape_a = [2usize, 3usize];
        let a = ndt_tensor_f64_zeros(shape_a.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(a, 1.0), NDT_SUCCESS);

        // Create B: 3x4 matrix of ones
        let shape_b = [3usize, 4usize];
        let b = ndt_tensor_f64_zeros(shape_b.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(b, 1.0), NDT_SUCCESS);

        // Create grad_output: 2x4 matrix of ones
        let shape_grad = [2usize, 4usize];
        let grad_output = ndt_tensor_f64_zeros(shape_grad.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(grad_output, 1.0), NDT_SUCCESS);

        // Labels
        let labels_a = [1i64, -1i64];
        let labels_b = [-1i64, 2i64];

        let mut grad_a: *mut ndt_tensor_f64 = ptr::null_mut();
        let mut grad_b: *mut ndt_tensor_f64 = ptr::null_mut();

        ndt_tensor_f64_contract_vjp(
            a,
            labels_a.as_ptr() as *const c_long,
            2,
            b,
            labels_b.as_ptr() as *const c_long,
            2,
            grad_output,
            &mut grad_a,
            &mut grad_b,
            &mut status,
        );

        assert_eq!(status, NDT_SUCCESS);
        assert!(!grad_a.is_null());
        assert!(!grad_b.is_null());

        // Check grad_a shape is 2x3
        assert_eq!(ndt_tensor_f64_ndim(grad_a), 2);
        let mut ga_shape = [0usize; 2];
        ndt_tensor_f64_shape(grad_a, ga_shape.as_mut_ptr());
        assert_eq!(ga_shape, [2, 3]);

        // Check grad_b shape is 3x4
        assert_eq!(ndt_tensor_f64_ndim(grad_b), 2);
        let mut gb_shape = [0usize; 2];
        ndt_tensor_f64_shape(grad_b, gb_shape.as_mut_ptr());
        assert_eq!(gb_shape, [3, 4]);

        // grad_a[i,j] = sum_k B[j,k] = 4 (since B is all ones and has 4 columns)
        let mut val = 0.0;
        assert_eq!(ndt_tensor_f64_get_linear(grad_a, 0, &mut val), NDT_SUCCESS);
        assert_eq!(val, 4.0);

        // grad_b[j,k] = sum_i A[i,j] = 2 (since A is all ones and has 2 rows)
        assert_eq!(ndt_tensor_f64_get_linear(grad_b, 0, &mut val), NDT_SUCCESS);
        assert_eq!(val, 2.0);

        ndt_tensor_f64_release(a);
        ndt_tensor_f64_release(b);
        ndt_tensor_f64_release(grad_output);
        ndt_tensor_f64_release(grad_a);
        ndt_tensor_f64_release(grad_b);
    }

    #[test]
    fn test_contract_jvp_null_tangents() {
        // Test with NULL tangents (both constants)
        let mut status: StatusCode = -999;

        // Create A: 2x3 matrix of ones
        let shape_a = [2usize, 3usize];
        let a = ndt_tensor_f64_zeros(shape_a.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(a, 1.0), NDT_SUCCESS);

        // Create B: 3x4 matrix of ones
        let shape_b = [3usize, 4usize];
        let b = ndt_tensor_f64_zeros(shape_b.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(b, 1.0), NDT_SUCCESS);

        // Labels
        let labels_a = [1i64, -1i64];
        let labels_b = [-1i64, 2i64];

        let mut primal: *mut ndt_tensor_f64 = ptr::null_mut();
        let mut tangent: *mut ndt_tensor_f64 = ptr::null_mut();

        ndt_tensor_f64_contract_jvp(
            a,
            labels_a.as_ptr() as *const c_long,
            2,
            b,
            labels_b.as_ptr() as *const c_long,
            2,
            ptr::null(), // tangent_a = NULL
            ptr::null(), // tangent_b = NULL
            &mut primal,
            &mut tangent,
            &mut status,
        );

        assert_eq!(status, NDT_SUCCESS);
        assert!(!primal.is_null());
        assert!(tangent.is_null()); // Should be NULL when both tangents are NULL

        // Check primal shape is 2x4
        assert_eq!(ndt_tensor_f64_ndim(primal), 2);
        let mut primal_shape = [0usize; 2];
        ndt_tensor_f64_shape(primal, primal_shape.as_mut_ptr());
        assert_eq!(primal_shape, [2, 4]);

        // Each element should be 3 (sum over contracted dimension of size 3)
        let mut val = 0.0;
        assert_eq!(ndt_tensor_f64_get_linear(primal, 0, &mut val), NDT_SUCCESS);
        assert_eq!(val, 3.0);

        ndt_tensor_f64_release(a);
        ndt_tensor_f64_release(b);
        ndt_tensor_f64_release(primal);
        // Don't release tangent since it's null
    }

    #[test]
    fn test_contract_jvp_with_tangent_a() {
        // Test with tangent only for A
        let mut status: StatusCode = -999;

        // Create A: 2x3 matrix of ones
        let shape_a = [2usize, 3usize];
        let a = ndt_tensor_f64_zeros(shape_a.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(a, 1.0), NDT_SUCCESS);

        // Create B: 3x4 matrix of ones
        let shape_b = [3usize, 4usize];
        let b = ndt_tensor_f64_zeros(shape_b.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(b, 1.0), NDT_SUCCESS);

        // Create tangent_a: 2x3 matrix of ones
        let tangent_a = ndt_tensor_f64_zeros(shape_a.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(tangent_a, 1.0), NDT_SUCCESS);

        // Labels
        let labels_a = [1i64, -1i64];
        let labels_b = [-1i64, 2i64];

        let mut primal: *mut ndt_tensor_f64 = ptr::null_mut();
        let mut tangent: *mut ndt_tensor_f64 = ptr::null_mut();

        ndt_tensor_f64_contract_jvp(
            a,
            labels_a.as_ptr() as *const c_long,
            2,
            b,
            labels_b.as_ptr() as *const c_long,
            2,
            tangent_a,   // tangent_a = ones
            ptr::null(), // tangent_b = NULL
            &mut primal,
            &mut tangent,
            &mut status,
        );

        assert_eq!(status, NDT_SUCCESS);
        assert!(!primal.is_null());
        assert!(!tangent.is_null());

        // Check tangent shape is 2x4
        assert_eq!(ndt_tensor_f64_ndim(tangent), 2);
        let mut tangent_shape = [0usize; 2];
        ndt_tensor_f64_shape(tangent, tangent_shape.as_mut_ptr());
        assert_eq!(tangent_shape, [2, 4]);

        // Tangent: dC = dA @ B = ones(2,3) @ ones(3,4) = 3 * ones(2,4)
        let mut val = 0.0;
        assert_eq!(ndt_tensor_f64_get_linear(tangent, 0, &mut val), NDT_SUCCESS);
        assert_eq!(val, 3.0);

        ndt_tensor_f64_release(a);
        ndt_tensor_f64_release(b);
        ndt_tensor_f64_release(tangent_a);
        ndt_tensor_f64_release(primal);
        ndt_tensor_f64_release(tangent);
    }

    #[test]
    fn test_contract_jvp_with_both_tangents() {
        // Test with tangents for both A and B
        let mut status: StatusCode = -999;

        // Create A: 2x3 matrix of ones
        let shape_a = [2usize, 3usize];
        let a = ndt_tensor_f64_zeros(shape_a.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(a, 1.0), NDT_SUCCESS);

        // Create B: 3x4 matrix of ones
        let shape_b = [3usize, 4usize];
        let b = ndt_tensor_f64_zeros(shape_b.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(b, 1.0), NDT_SUCCESS);

        // Create tangent_a: 2x3 matrix of ones
        let tangent_a = ndt_tensor_f64_zeros(shape_a.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(tangent_a, 1.0), NDT_SUCCESS);

        // Create tangent_b: 3x4 matrix of ones
        let tangent_b = ndt_tensor_f64_zeros(shape_b.as_ptr(), 2, &mut status);
        assert_eq!(ndt_tensor_f64_fill(tangent_b, 1.0), NDT_SUCCESS);

        // Labels
        let labels_a = [1i64, -1i64];
        let labels_b = [-1i64, 2i64];

        let mut primal: *mut ndt_tensor_f64 = ptr::null_mut();
        let mut tangent: *mut ndt_tensor_f64 = ptr::null_mut();

        ndt_tensor_f64_contract_jvp(
            a,
            labels_a.as_ptr() as *const c_long,
            2,
            b,
            labels_b.as_ptr() as *const c_long,
            2,
            tangent_a,
            tangent_b,
            &mut primal,
            &mut tangent,
            &mut status,
        );

        assert_eq!(status, NDT_SUCCESS);
        assert!(!primal.is_null());
        assert!(!tangent.is_null());

        // Tangent: dC = dA @ B + A @ dB = 3 + 3 = 6
        let mut val = 0.0;
        assert_eq!(ndt_tensor_f64_get_linear(tangent, 0, &mut val), NDT_SUCCESS);
        assert_eq!(val, 6.0);

        ndt_tensor_f64_release(a);
        ndt_tensor_f64_release(b);
        ndt_tensor_f64_release(tangent_a);
        ndt_tensor_f64_release(tangent_b);
        ndt_tensor_f64_release(primal);
        ndt_tensor_f64_release(tangent);
    }

    #[test]
    fn test_tensor_ones() {
        let shape = [2usize, 3usize];
        let mut status: StatusCode = -999;

        let tensor = ndt_tensor_f64_ones(shape.as_ptr(), 2, &mut status);
        assert_eq!(status, NDT_SUCCESS);
        assert!(!tensor.is_null());

        assert_eq!(ndt_tensor_f64_ndim(tensor), 2);
        assert_eq!(ndt_tensor_f64_len(tensor), 6);

        // Verify all values are 1.0
        for i in 0..6 {
            let mut val = 0.0;
            assert_eq!(ndt_tensor_f64_get_linear(tensor, i, &mut val), NDT_SUCCESS);
            assert_eq!(val, 1.0);
        }

        ndt_tensor_f64_release(tensor);
    }

    #[test]
    fn test_tensor_ones_scalar() {
        // 0-dimensional tensor (scalar)
        let mut status: StatusCode = -999;
        let tensor = ndt_tensor_f64_ones(std::ptr::null(), 0, &mut status);
        assert_eq!(status, NDT_SUCCESS);
        assert!(!tensor.is_null());
        assert_eq!(ndt_tensor_f64_len(tensor), 1);

        let mut val = 0.0;
        assert_eq!(ndt_tensor_f64_get_linear(tensor, 0, &mut val), NDT_SUCCESS);
        assert_eq!(val, 1.0);

        ndt_tensor_f64_release(tensor);
    }

    #[test]
    fn test_tensor_rand() {
        let shape = [2usize, 3usize];
        let mut status: StatusCode = -999;

        let tensor = ndt_tensor_f64_rand(shape.as_ptr(), 2, &mut status);
        assert_eq!(status, NDT_SUCCESS);
        assert!(!tensor.is_null());

        assert_eq!(ndt_tensor_f64_ndim(tensor), 2);
        assert_eq!(ndt_tensor_f64_len(tensor), 6);

        // Verify all values are in [0, 1)
        for i in 0..6 {
            let mut val = 0.0;
            assert_eq!(ndt_tensor_f64_get_linear(tensor, i, &mut val), NDT_SUCCESS);
            assert!((0.0..1.0).contains(&val), "value {} not in [0, 1)", val);
        }

        ndt_tensor_f64_release(tensor);
    }

    #[test]
    fn test_tensor_rand_null_status() {
        let shape = [2usize, 3usize];
        let tensor = ndt_tensor_f64_rand(shape.as_ptr(), 2, std::ptr::null_mut());
        assert!(tensor.is_null());
    }

    #[test]
    fn test_tensor_randn() {
        let shape = [100usize]; // Use larger size for statistical test
        let mut status: StatusCode = -999;

        let tensor = ndt_tensor_f64_randn(shape.as_ptr(), 1, &mut status);
        assert_eq!(status, NDT_SUCCESS);
        assert!(!tensor.is_null());

        assert_eq!(ndt_tensor_f64_ndim(tensor), 1);
        assert_eq!(ndt_tensor_f64_len(tensor), 100);

        // Verify values are roughly normal (mean near 0)
        let data = ndt_tensor_f64_data(tensor);
        assert!(!data.is_null());

        let sum: f64 = unsafe { std::slice::from_raw_parts(data, 100).iter().sum() };
        let mean = sum / 100.0;
        assert!(
            mean.abs() < 0.5,
            "mean {} too far from 0 for normal distribution",
            mean
        );

        ndt_tensor_f64_release(tensor);
    }

    #[test]
    fn test_tensor_randn_null_shape_with_ndim() {
        let mut status: StatusCode = -999;
        let tensor = ndt_tensor_f64_randn(std::ptr::null(), 2, &mut status);
        assert_eq!(status, NDT_INVALID_ARGUMENT);
        assert!(tensor.is_null());
    }
}
