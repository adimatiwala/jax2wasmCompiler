# jax2wasmCompiler

A toolchain for converting JAX Python functions to WebAssembly (WASM) via HLO (High Level Optimizer) and WAT (WebAssembly Text) intermediate representations.

## Features
- Converts JAX functions to HLO IR
- Converts HLO IR to WAT (WebAssembly Text)
- Converts WAT to WASM (WebAssembly binary) internally using Rust
- Handles both forward and gradient computations

## Requirements
- Python 3.8+
- Rust (with Cargo)
- [pip](https://pip.pypa.io/en/stable/)

## Python Dependencies
Install Python dependencies with:
```sh
pip install -r requirements.txt
```

## Rust Dependencies
All Rust dependencies are managed via Cargo. The project uses the [`wat`](https://crates.io/crates/wat) crate for internal WAT-to-WASM conversion.

## Setup
1. **Clone the repository:**
   ```sh
git clone <repo-url>
cd jax2wasmCompiler
```
2. **Install Python dependencies:**
   ```sh
pip install -r requirements.txt
```
3. **Build Rust project:**
   ```sh
cargo build --release
```

## Usage

### 1. Generate HLO from JAX
Run the Python script to generate HLO files for both the forward and gradient passes:
```sh
python3 src/jax2hlo.py
```
This will create `src/forward_optimized.hlo` and `src/grad_optimized.hlo`.

### 2. Compile HLO to WASM
Run the Rust pipeline to convert HLO to WAT and then to WASM:
```sh
cargo run --release
```
This will generate:
- `target/forward_generated.wat` and `target/forward_generated.wasm`
- `target/grad_generated.wat` and `target/grad_generated.wasm`

### 3. (Optional) Run or test the generated WASM
You can use any WASM runtime or JS environment to load and test the generated `.wasm` files.

## Project Structure
- `src/jax2hlo.py` — Converts JAX functions to HLO IR
- `src/` — Rust source code for HLO parsing, WAT/WASM generation
- `target/` — Output directory for generated WAT/WASM files
- `requirements.txt` — Python dependencies
- `Cargo.toml` — Rust dependencies

## Extending
- Add more HLO op support in the Rust code as needed
- Improve memory management and type support
- Add more tests and examples

## License
MIT
