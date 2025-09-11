#!/bin/sh
#
# It sets the LLVM environment variables.
#
# You can copy this file to .envrc/.env and adapt it for your environment.

# This line will ensure that the script can be used from any directory, not just
# the project's root.
ROOT_DIR="$(realpath $(dirname "${BASH_SOURCE[0]}"))"

case $(uname) in
  Darwin)
    # If installed with brew
    LIBRARY_PATH=/opt/homebrew/lib
    MLIR_SYS_200_PREFIX="$(brew --prefix llvm@20)"
    LLVM_SYS_201_PREFIX="$(brew --prefix llvm@20)"
    TABLEGEN_200_PREFIX="$(brew --prefix llvm@20)"

    export LIBRARY_PATH
    export MLIR_SYS_200_PREFIX
    export LLVM_SYS_201_PREFIX
    export TABLEGEN_200_PREFIX
  ;;
  Linux)
    # If installed from Debian/Ubuntu repository:
    MLIR_SYS_200_PREFIX=/usr/lib/llvm-20
    LLVM_SYS_201_PREFIX=/usr/lib/llvm-20
    TABLEGEN_200_PREFIX=/usr/lib/llvm-20

    export MLIR_SYS_200_PREFIX
    export LLVM_SYS_201_PREFIX
    export TABLEGEN_200_PREFIX
  ;;
esac
