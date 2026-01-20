"""Library loading and FFI declarations for ndtensors-rs."""

import ctypes
import platform
from pathlib import Path
from typing import Optional

# Global library instance
_lib: Optional[ctypes.CDLL] = None


def _get_lib_extension() -> str:
    """Get the shared library extension for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return ".dylib"
    elif system == "Linux":
        return ".so"
    elif system == "Windows":
        return ".dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def _get_lib_path() -> Path:
    """Locate the shared library.

    Search order:
    1. Package directory (for installed package)
    2. Project target/release directory (for development)
    """
    ext = _get_lib_extension()
    lib_name = f"libndtensors_capi{ext}"

    # Check package directory first
    package_dir = Path(__file__).parent
    lib_path = package_dir / lib_name
    if lib_path.exists():
        return lib_path

    # Check project target/release directory (development mode)
    project_root = package_dir.parent.parent.parent.parent
    lib_path = project_root / "target" / "release" / lib_name
    if lib_path.exists():
        return lib_path

    # Also check debug build
    lib_path = project_root / "target" / "debug" / lib_name
    if lib_path.exists():
        return lib_path

    raise FileNotFoundError(
        f"Could not find {lib_name}. "
        "Please run 'cargo build --release -p ndtensors-capi' first."
    )


def _declare_functions(lib: ctypes.CDLL) -> None:
    """Declare function signatures for type safety."""
    # ==========================================================================
    # Tensor creation functions
    # ==========================================================================

    # ndt_tensor_f64_zeros
    lib.ndt_tensor_f64_zeros.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),  # shape
        ctypes.c_size_t,  # ndim
        ctypes.POINTER(ctypes.c_int),  # status
    ]
    lib.ndt_tensor_f64_zeros.restype = ctypes.c_void_p

    # ndt_tensor_f64_ones
    lib.ndt_tensor_f64_ones.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),  # shape
        ctypes.c_size_t,  # ndim
        ctypes.POINTER(ctypes.c_int),  # status
    ]
    lib.ndt_tensor_f64_ones.restype = ctypes.c_void_p

    # ndt_tensor_f64_rand
    lib.ndt_tensor_f64_rand.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),  # shape
        ctypes.c_size_t,  # ndim
        ctypes.POINTER(ctypes.c_int),  # status
    ]
    lib.ndt_tensor_f64_rand.restype = ctypes.c_void_p

    # ndt_tensor_f64_randn
    lib.ndt_tensor_f64_randn.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),  # shape
        ctypes.c_size_t,  # ndim
        ctypes.POINTER(ctypes.c_int),  # status
    ]
    lib.ndt_tensor_f64_randn.restype = ctypes.c_void_p

    # ndt_tensor_f64_from_data
    lib.ndt_tensor_f64_from_data.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # data
        ctypes.c_size_t,  # len
        ctypes.POINTER(ctypes.c_size_t),  # shape
        ctypes.c_size_t,  # ndim
        ctypes.POINTER(ctypes.c_int),  # status
    ]
    lib.ndt_tensor_f64_from_data.restype = ctypes.c_void_p

    # ==========================================================================
    # Tensor lifecycle functions
    # ==========================================================================

    # ndt_tensor_f64_release
    lib.ndt_tensor_f64_release.argtypes = [ctypes.c_void_p]
    lib.ndt_tensor_f64_release.restype = None

    # ndt_tensor_f64_clone
    lib.ndt_tensor_f64_clone.argtypes = [ctypes.c_void_p]
    lib.ndt_tensor_f64_clone.restype = ctypes.c_void_p

    # ==========================================================================
    # Tensor query functions
    # ==========================================================================

    # ndt_tensor_f64_ndim
    lib.ndt_tensor_f64_ndim.argtypes = [ctypes.c_void_p]
    lib.ndt_tensor_f64_ndim.restype = ctypes.c_size_t

    # ndt_tensor_f64_len
    lib.ndt_tensor_f64_len.argtypes = [ctypes.c_void_p]
    lib.ndt_tensor_f64_len.restype = ctypes.c_size_t

    # ndt_tensor_f64_shape
    lib.ndt_tensor_f64_shape.argtypes = [
        ctypes.c_void_p,  # tensor
        ctypes.POINTER(ctypes.c_size_t),  # out
    ]
    lib.ndt_tensor_f64_shape.restype = ctypes.c_int

    # ndt_tensor_f64_data
    lib.ndt_tensor_f64_data.argtypes = [ctypes.c_void_p]
    lib.ndt_tensor_f64_data.restype = ctypes.POINTER(ctypes.c_double)

    # ==========================================================================
    # Element access functions
    # ==========================================================================

    # ndt_tensor_f64_get_linear
    lib.ndt_tensor_f64_get_linear.argtypes = [
        ctypes.c_void_p,  # tensor
        ctypes.c_size_t,  # index
        ctypes.POINTER(ctypes.c_double),  # out
    ]
    lib.ndt_tensor_f64_get_linear.restype = ctypes.c_int

    # ndt_tensor_f64_set_linear
    lib.ndt_tensor_f64_set_linear.argtypes = [
        ctypes.c_void_p,  # tensor
        ctypes.c_size_t,  # index
        ctypes.c_double,  # value
    ]
    lib.ndt_tensor_f64_set_linear.restype = ctypes.c_int

    # ndt_tensor_f64_fill
    lib.ndt_tensor_f64_fill.argtypes = [
        ctypes.c_void_p,  # tensor
        ctypes.c_double,  # value
    ]
    lib.ndt_tensor_f64_fill.restype = ctypes.c_int

    # ==========================================================================
    # Tensor operations
    # ==========================================================================

    # ndt_tensor_f64_permutedims
    lib.ndt_tensor_f64_permutedims.argtypes = [
        ctypes.c_void_p,  # tensor
        ctypes.POINTER(ctypes.c_size_t),  # perm
        ctypes.c_size_t,  # ndim
        ctypes.POINTER(ctypes.c_int),  # status
    ]
    lib.ndt_tensor_f64_permutedims.restype = ctypes.c_void_p

    # ndt_tensor_f64_contract
    lib.ndt_tensor_f64_contract.argtypes = [
        ctypes.c_void_p,  # a
        ctypes.POINTER(ctypes.c_long),  # labels_a
        ctypes.c_size_t,  # ndim_a
        ctypes.c_void_p,  # b
        ctypes.POINTER(ctypes.c_long),  # labels_b
        ctypes.c_size_t,  # ndim_b
        ctypes.POINTER(ctypes.c_int),  # status
    ]
    lib.ndt_tensor_f64_contract.restype = ctypes.c_void_p

    # ndt_tensor_f64_contract_vjp
    lib.ndt_tensor_f64_contract_vjp.argtypes = [
        ctypes.c_void_p,  # a
        ctypes.POINTER(ctypes.c_long),  # labels_a
        ctypes.c_size_t,  # ndim_a
        ctypes.c_void_p,  # b
        ctypes.POINTER(ctypes.c_long),  # labels_b
        ctypes.c_size_t,  # ndim_b
        ctypes.c_void_p,  # grad_output
        ctypes.POINTER(ctypes.c_void_p),  # grad_a_out
        ctypes.POINTER(ctypes.c_void_p),  # grad_b_out
        ctypes.POINTER(ctypes.c_int),  # status
    ]
    lib.ndt_tensor_f64_contract_vjp.restype = None


def _load_library() -> ctypes.CDLL:
    """Load the ndtensors_capi shared library."""
    lib_path = _get_lib_path()
    lib = ctypes.CDLL(str(lib_path))
    _declare_functions(lib)
    return lib


def get_lib() -> ctypes.CDLL:
    """Get the loaded library instance (singleton)."""
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib
