# Cairo Native
[![test](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml/badge.svg)](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml)
[![mdbook](https://img.shields.io/badge/mdbook-link-blue)](https://lambdaclass.github.io/cairo_native/)

A compiler to convert Cairo's intermediate representation "Sierra" code to machine code via MLIR and LLVM.

## Documentation

There is an mdbook in the `docs` folder. Build and read it with
```bash
make book
```

## Dependencies
- mdbook
- LLVM 16+ with MLIR
- Rust

## Setup

Install mdbook and other documentation dependencies:
```bash
cargo install mdbook mdbook-toc mdbook-mermaid
```

Install LLVM with MLIR. You can use the official packages provided by LLVM.

### Linux

Setup a environment variable called `MLIR_SYS_160_PREFIX` pointing to the llvm directory:

```bash
MLIR_SYS_160_PREFIX=/usr/lib/llvm-16
```

### MacOS
```bash
brew install llvm@16
export MLIR_SYS_160_PREFIX=/opt/homebrew/opt/llvm@16
```

## CLI Interface

sierra2mlir:
```
Usage: sierra2mlir [OPTIONS] <INPUT>

Arguments:
  <INPUT>

Options:
  -o, --output <OUTPUT>  [default: -]
  -h, --help             Print help
```

sierrajit:
```
Usage: sierrajit [OPTIONS] <INPUT> <ENTRY_POINT>

Arguments:
  <INPUT>
  <ENTRY_POINT>

Options:
  -i, --inputs <INPUTS>
  -o, --outputs <OUTPUTS>
  -p, --print-outputs
  -g, --available-gas <AVAILABLE_GAS>
  -h, --help                           Print help
```

## Benchmarking

### Requirements
- [hyperfine](https://github.com/sharkdp/hyperfine): `cargo install hyperfine`
- [cairo >=1.0](https://github.com/starkware-libs/cairo)
- LLVM 16

You need to setup some environment variables:
```bash
$MLIR_SYS_160_PREFIX=/path/to/llvm16  # Required for non-standard LLVM install locations.
```

```bash
make bench
```

## MLIR Resources
- https://mlir.llvm.org/docs/Tutorials/

## From MLIR to native binary
```bash
# to mlir with llvm dialect
cargo r --release --features build.cli --bin sierra2mlir -- program.sierra -o program.mlir

# translate mlir to llvm-ir
"$MLIR_SYS_160_PREFIX"/bin/mlir-translate --mlir-to-llvmir program.mlir -o program.ll

# compile natively
"$MLIR_SYS_160_PREFIX"/bin/clang program.ll -Wno-override-module \
    -L "$MLIR_SYS_160_PREFIX"/lib -L"./target/release/" \
    -lsierra2mlir_utils -lmlir_c_runner_utils \
    -Wl,-rpath "$MLIR_SYS_160_PREFIX"/lib \
    -Wl,-rpath ./target/release/ \
    -o program

./program
```
