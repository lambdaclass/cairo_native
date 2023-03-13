#!/usr/bin/env bash

# Requires MLIR compiled with -DMLIR_ENABLE_CUDA_RUNNER=ON

# configure the mlir directory here
MLIR_DIR=~/data/mlir

MLIR_OPT=$MLIR_DIR/bin/mlir-opt
MLIR_CPU_RUNNER=$MLIR_DIR/bin/mlir-cpu-runner
CUDA_MLIR_RUNTIME="$MLIR_DIR/lib/libmlir_cuda_runtime.so"

echo "input is: $1"

input_filename=$(basename -- "$1")

extension="${input_filename##*.}"
filename="${input_filename%.*}"
OUTPUT_MLIR="${filename}-translated.mlir"

echo "translating base mlir to llvm dialect"
$MLIR_OPT --convert-gpu-to-nvvm --test-gpu-to-cubin --gpu-to-cubin --gpu-to-llvm $1 > $OUTPUT_MLIR
echo "saved to $OUTPUT_MLIR"

echo "runner output:"
$MLIR_CPU_RUNNER --entry-point-result=i32  --shared-libs="$CUDA_MLIR_RUNTIME" $OUTPUT_MLIR
