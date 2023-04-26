# Setup the project

First clone the repository:

```bash
git clone git@github.com:lambdaclass/cairo_sierra_2_MLIR.git
```

## Get the dependencies

You will need:

- LLVM 16, with MLIR project enabled
- Rust
- mdbook: if you want to build this book.

## About LLVM

If your system doesn't have LLVM 16 available, you can point the build script to it by setting up the environment variable `MLIR_SYS_160_PREFIX`

For example:

```bash
MLIR_SYS_160_PREFIX=/usr/lib/llvm/16
```

## Use make

With make you can make your life easier, we got some common targets set up:

```bash
# first build the project
make build

# build the example programs
make compile-mlir

# build the example programs with optimizations on
make compile-mlir-opt

# run tests
make test

# benchmark
make bench

# clean
make clean
```
