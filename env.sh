#!/bin/sh
#
# It sets the LLVM environment variables.
#
# You can copy this file to .envrc/.env and adapt it for your environment.

case $(uname) in
  Darwin)
    # If installed with brew
    LIBRARY_PATH=/opt/homebrew/lib
    MLIR_SYS_180_PREFIX="$(brew --prefix llvm@18)"
    LLVM_SYS_180_PREFIX="$(brew --prefix llvm@18)"
    TABLEGEN_180_PREFIX="$(brew --prefix llvm@18)"
    CAIRO_NATIVE_RUNTIME_LIBDIR="$(pwd)/target/debug"

    export LIBRARY_PATH
    export MLIR_SYS_180_PREFIX
    export LLVM_SYS_180_PREFIX
    export TABLEGEN_180_PREFIX
    export CAIRO_NATIVE_RUNTIME_LIBDIR
  ;;
  Linux)
    # If installed from Debian/Ubuntu repository:
    MLIR_SYS_180_PREFIX=/usr/lib/llvm-18
    LLVM_SYS_180_PREFIX=/usr/lib/llvm-18
    TABLEGEN_180_PREFIX=/usr/lib/llvm-18
    CAIRO_NATIVE_RUNTIME_LIBDIR="$(pwd)/target/debug"

    export MLIR_SYS_180_PREFIX
    export LLVM_SYS_180_PREFIX
    export TABLEGEN_180_PREFIX
    export CAIRO_NATIVE_RUNTIME_LIBDIR
  ;;
esac
