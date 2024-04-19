#!/usr/bin/env bash

# This script is only useful on macOS using brew.
# It sets the LLVM environment variables.
export LIBRARY_PATH=/opt/homebrew/lib
MLIR_SYS_170_PREFIX="$(brew --prefix llvm)"
LLVM_SYS_170_PREFIX="$(brew --prefix llvm)"
TABLEGEN_170_PREFIX="$(brew --prefix llvm)"
CAIRO_NATIVE_RUNTIME_LIBDIR="$(pwd)/target/debug"

export MLIR_SYS_170_PREFIX
export LLVM_SYS_170_PREFIX
export TABLEGEN_170_PREFIX
export CAIRO_NATIVE_RUNTIME_LIBDIR
