#!/usr/bin/env bash

# This script is only useful on macOS using brew.
# It sets the LLVM environment variables.

MLIR_SYS_170_PREFIX="$(brew --prefix llvm@17)"
LLVM_SYS_170_PREFIX="$(brew --prefix llvm@17)"
TABLEGEN_170_PREFIX="$(brew --prefix llvm@17)"
CAIRO_NATIVE_RUNTIME_LIBDIR="$(pwd)/target/debug"

export MLIR_SYS_170_PREFIX
export LLVM_SYS_170_PREFIX
export TABLEGEN_170_PREFIX
export CAIRO_NATIVE_RUNTIME_LIBDIR
