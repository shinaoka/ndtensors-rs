//! C API for ndtensors
//!
//! This crate provides a C-compatible interface to the ndtensors library,
//! allowing it to be called from Julia, Python, C, and other languages.

use libc::{c_double, c_int, size_t};
use ndtensors::Tensor;
use std::panic::catch_unwind;
use std::ptr;

// Status codes
pub type StatusCode = c_int;

pub const NDT_SUCCESS: StatusCode = 0;
pub const NDT_INVALID_ARGUMENT: StatusCode = -1;
pub const NDT_SHAPE_MISMATCH: StatusCode = -2;
pub const NDT_INDEX_OUT_OF_BOUNDS: StatusCode = -3;
pub const NDT_INTERNAL_ERROR: StatusCode = -4;

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
        assert_eq!(ndt_tensor_f64_shape(tensor, out_shape.as_mut_ptr()), NDT_SUCCESS);
        assert_eq!(out_shape, [2, 3]);

        ndt_tensor_f64_release(tensor);
    }

    #[test]
    fn test_tensor_from_data() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f64];
        let shape = [2usize, 3usize];
        let mut status: StatusCode = -999;

        let tensor = ndt_tensor_f64_from_data(
            data.as_ptr(),
            data.len(),
            shape.as_ptr(),
            2,
            &mut status,
        );
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
}
