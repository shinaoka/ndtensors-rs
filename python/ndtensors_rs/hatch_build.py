"""Custom build hook for building and copying the Rust library."""

import platform
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class RustBuildHook(BuildHookInterface):
    """Build hook that compiles the Rust library and copies it to the package."""

    PLUGIN_NAME = "rust-build"

    def initialize(self, version: str, build_data: dict) -> None:
        """Build Rust library and copy to package before building wheel."""
        # Find project root (two levels up from this file's directory)
        hook_dir = Path(__file__).parent
        project_root = hook_dir.parent.parent

        # Build Rust library
        print("Building Rust library...")
        subprocess.check_call(
            ["cargo", "build", "--release", "-p", "ndtensors-capi"],
            cwd=project_root,
        )

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

        # Copy library to package
        src = project_root / "target" / "release" / lib_name
        dst = hook_dir / "src" / "ndtensors_rs" / lib_name

        if not src.exists():
            raise FileNotFoundError(f"Built library not found at {src}")

        print(f"Copying {src} to {dst}")
        shutil.copy(src, dst)

        # Include the library in the wheel
        build_data["force_include"][str(dst)] = f"ndtensors_rs/{lib_name}"
