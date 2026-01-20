#!/bin/bash
set -euo pipefail

# Run Python tests for ndtensors-rs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_PKG_DIR="$PROJECT_ROOT/python/ndtensors_rs"

echo "=== Building Rust library ==="
cargo build --release -p ndtensors-capi

echo "=== Copying shared library to package ==="
# Determine library extension based on OS
case "$(uname -s)" in
    Darwin*)  LIB_EXT=".dylib" ;;
    Linux*)   LIB_EXT=".so" ;;
    MINGW*|MSYS*|CYGWIN*)  LIB_EXT=".dll" ;;
    *)        echo "Unsupported OS"; exit 1 ;;
esac

LIB_NAME="libndtensors_capi${LIB_EXT}"
cp "$PROJECT_ROOT/target/release/$LIB_NAME" "$PYTHON_PKG_DIR/src/ndtensors_rs/"

echo "=== Setting up Python environment ==="
cd "$PYTHON_PKG_DIR"

# Create virtual environment and install with uv
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

echo "=== Running Python tests ==="
pytest -v

echo "=== All Python tests passed ==="
