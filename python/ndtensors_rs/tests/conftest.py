"""Pytest configuration for ndtensors-rs tests."""

import platform
import shutil
import subprocess
from pathlib import Path


def pytest_configure(config):
    """Build Rust library before running tests if needed."""
    # Find project root
    tests_dir = Path(__file__).parent
    pkg_dir = tests_dir.parent
    project_root = pkg_dir.parent.parent

    # Determine library name based on platform
    system = platform.system()
    if system == "Darwin":
        lib_name = "libndtensors_capi.dylib"
    elif system == "Linux":
        lib_name = "libndtensors_capi.so"
    elif system == "Windows":
        lib_name = "ndtensors_capi.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    # Check if library exists in package
    lib_in_pkg = pkg_dir / "src" / "ndtensors_rs" / lib_name
    lib_in_target = project_root / "target" / "release" / lib_name

    # Build if library doesn't exist or is older than source
    needs_build = not lib_in_pkg.exists()
    if not needs_build and lib_in_target.exists():
        needs_build = lib_in_target.stat().st_mtime > lib_in_pkg.stat().st_mtime

    if needs_build:
        print(f"\n=== Building Rust library ===")
        subprocess.check_call(
            ["cargo", "build", "--release", "-p", "ndtensors-capi"],
            cwd=project_root,
        )

        print(f"=== Copying {lib_name} to package ===")
        shutil.copy(lib_in_target, lib_in_pkg)
