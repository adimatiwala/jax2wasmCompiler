#!/bin/bash
set -e

echo "Running JAX to HLO demo..."
python3 src/jax2hlo.py --demo

echo "Running Rust pipeline (HLO to WASM)..."
cargo run --release

echo "Demo complete!"
echo "Check target/forward_generated.wat, target/forward_generated.wasm, etc."
