"""Setup script for ndtensors-rs Python bindings.

This script handles building the Rust library and copying it to the package.
"""

import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


def get_lib_extension() -> str:
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


def build_rust_library() -> Path:
    """Build the Rust library and return the path to the built library."""
    # Find the project root (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent.resolve()

    # Build Rust library
    print(f"Building Rust library in {project_root}...")
    subprocess.check_call(
        ["cargo", "build", "--release", "-p", "ndtensors-capi"],
        cwd=project_root,
    )

    # Return path to built library
    ext = get_lib_extension()
    lib_name = f"libndtensors_capi{ext}"
    lib_path = project_root / "target" / "release" / lib_name

    if not lib_path.exists():
        raise FileNotFoundError(f"Built library not found at {lib_path}")

    return lib_path


def copy_library_to_package(lib_path: Path) -> None:
    """Copy the built library to the package directory."""
    package_dir = Path(__file__).parent / "src" / "ndtensors_rs"
    dest_path = package_dir / lib_path.name

    print(f"Copying {lib_path} to {dest_path}...")
    shutil.copy(lib_path, dest_path)


class BuildRustAndPy(build_py):
    """Custom build command that builds Rust library first."""

    def run(self):
        lib_path = build_rust_library()
        copy_library_to_package(lib_path)
        super().run()


class DevelopWithRust(develop):
    """Custom develop command that builds Rust library first."""

    def run(self):
        lib_path = build_rust_library()
        copy_library_to_package(lib_path)
        super().run()


setup(
    cmdclass={
        "build_py": BuildRustAndPy,
        "develop": DevelopWithRust,
    }
)
