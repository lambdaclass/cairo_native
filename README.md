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

If you've compiled LLVM manually, or installed it in a non-standard path, then please set the
environment variable `MLIR_SYS_160_PREFIX` accordingly.

To build and install LLVM manually see the `scripts/install-llvm.sh` script.

Setup a environment variable called `MLIR_SYS_160_PREFIX` pointing to the mlir directory:

```bash
MLIR_SYS_160_PREFIX=~/mlir
```

### MacOS
```bash
brew install llvm@16
export MLIR_SYS_160_PREFIX=/opt/homebrew/opt/llvm@16
```

## CLI Interface
```
Usage: cli --input <INPUT> <COMMAND>

Commands:
  compile  Compile to MLIR with LLVM dialect, ready to be converted by `mlir-translate --mlir-to-llvmir`
  run      Compile and run a program. The entry point must be a function without arguments
  help     Print this message or the help of the given subcommand(s)

Options:
  -i, --input <INPUT>  The input sierra file
  -h, --help           Print help (see more with '--help')
  -V, --version        Print version
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
cargo r -- compile program.sierra -m --available-gas 9000000000 -o program.mlir

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
