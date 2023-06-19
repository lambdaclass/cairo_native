# Cairo Sierra to MLIR compiler
[![test](https://github.com/lambdaclass/cairo_sierra_to_mlir/actions/workflows/ci.yml/badge.svg)](https://github.com/lambdaclass/cairo_sierra_to_mlir/actions/workflows/ci.yml)
[![mdbook](https://img.shields.io/badge/mdbook-link-blue)](https://lambdaclass.github.io/cairo_sierra_2_MLIR/)

A compiler to convert Cairo's intermediate representation "Sierra" code to machine code via MLIR and LLVM.
The bindings used by the project will be our MLIR Rust binding called [mithril-oxide](https://github.com/lambdaclass/mithril-oxide).
test
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

To build LLVM manually, follow this steps:

```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.0/llvm-project-16.0.0.src.tar.xz

mkdir ~/mlir
tar -xf llvm-project-16.0.0.src.tar.xz
cd llvm-project-16.0.0.src.tar
mkdir build
cd build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_INSTALL_PREFIX=~/mlir
```

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
