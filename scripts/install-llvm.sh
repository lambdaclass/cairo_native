#!/usr/bin/env bash

# Exit after enconuntering an error.
set -e


mkdir -p llvm/dist
mkdir -p llvm/source

wget -P llvm/source "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.5/llvm-project-16.0.5.src.tar.xz"
tar -xf "llvm/source/llvm-project-16.0.5.src.tar.xz" -C llvm/source/
cd "llvm/source/llvm-project-16.0.5.src/"

mkdir -p build/
cd build/

cmake ../llvm \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="../../../dist/" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;lldb" \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="X86;AArch64"

cmake --build . -j$(nproc)
cmake --install .

echo "Please set MLIR_SYS_160_PREFIX to $(realpath ../../../dist/)"
