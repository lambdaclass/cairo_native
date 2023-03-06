# Cairo Sierra to MLIR compiler
[![test](https://github.com/lambdaclass/cairo_sierra_to_mlir/actions/workflows/test.yml/badge.svg)](https://github.com/lambdaclass/cairo_sierra_to_mlir/actions/workflows/test.yml)

A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR.

## Dependencies

- LLVM 16+ with MLIR
- Rust

## Project Layout

This project consists of two crates:

- mlir: Safe rusty bindings around the llvm-mlir-sys crate. Aims to provide safe easy to use MLIR bindings.
- sierra2mlir: The actual compiler, using the mlir crate.

## Setup

Check out the setup guide on the [llvm-mlir-sys](https://github.com/lambdaclass/llvm-mlir-sys) crate, which is the biggest hurdle to getting this running.

## MLIR Resources
- https://mlir.llvm.org/docs/Tutorials/
