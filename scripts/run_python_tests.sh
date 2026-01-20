#!/bin/bash
set -euo pipefail

# Run Python tests for ndtensors-rs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_PKG_DIR="$PROJECT_ROOT/python/ndtensors_rs"

echo "=== Building Rust library ==="
cargo build --release -p ndtensors-capi

echo "=== Setting up Python environment ==="
cd "$PYTHON_PKG_DIR"

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install package with dev dependencies
uv pip install -e ".[dev]"

echo "=== Running Python tests ==="
pytest -v

echo "=== All Python tests passed ==="
