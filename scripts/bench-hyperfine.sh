#!/usr/bin/env bash

# Configuration.
ROOT_DIR="$(dirname $(dirname ${0%/*}))"
MLIR_DIR="$MLIR_SYS_160_PREFIX"

CAIRO_SRCS=$(find \
    "$ROOT_DIR/programs/benches" \
    -type f -name "*.cairo")
IFS=$'\n' read -rd '' -a CAIRO_SRCS <<<"$CAIRO_SRCS"

CAIRO_RUN="${CAIRO_RUN:=cairo-run}"
COMPILER_CLI="$ROOT_DIR/target/release/sierra2mlir"
JIT_CLI="$ROOT_DIR/target/release/sierrajit"
OUTPUT_DIR="$ROOT_DIR/target/bench-outputs"

# Initial setup.
if [[ ! -e "$COMPILER_CLI" ]]
then
    echo "Compiler CLI not found. Is the project built in release mode?"
    exit 1
fi
if [[ -z "$MLIR_DIR" ]]
then
    echo "MLIR_DIR is empty. Did you forget to set MLIR_SYS_160_PREFIX?"
    exit 1
fi

set -e
mkdir -p "$OUTPUT_DIR"

# Benchmarking code.
run_bench() {
    base_path="${1%.cairo}"
    base_name=$(basename $base_path)

    >&2 echo "Benchmarking $1..."

    "$COMPILER_CLI" \
        "$base_path.cairo" \
        --output "$OUTPUT_DIR/$base_name.mlir" \
        >> /dev/stderr

    "$MLIR_DIR/bin/mlir-opt" \
        --canonicalize \
        --convert-scf-to-cf \
        --canonicalize \
        --cse \
        --expand-strided-metadata \
        --convert-memref-to-llvm \
        --convert-func-to-llvm \
        --convert-index-to-llvm \
        --reconcile-unrealized-casts \
        "$OUTPUT_DIR/$base_name.mlir" \
        -o "$OUTPUT_DIR/$base_name.opt.mlir" \
        >> /dev/stderr

    "$MLIR_DIR/bin/mlir-translate" \
        --mlir-to-llvmir \
        "$OUTPUT_DIR/$base_name.opt.mlir" \
        -o "$OUTPUT_DIR/$base_name.ll" \
        >> /dev/stderr

    "$MLIR_DIR/bin/clang" \
        -O3 \
        -Wno-override-module \
        "$base_path.c" \
        "$OUTPUT_DIR/$base_name.ll" \
        -L "target/release" \
        -Wl,-rpath "$MLIR_DIR/lib" \
        -Wl,-rpath "target/release" \
        -o "$OUTPUT_DIR/$base_name" \
        >> /dev/stderr

    "$MLIR_DIR/bin/clang" \
        -O3 \
        -march=native \
        -mtune=native \
        -Wno-override-module \
        "$base_path.c" \
        "$OUTPUT_DIR/$base_name.ll" \
        -L "target/release" \
        -Wl,-rpath "$MLIR_DIR/lib" \
        -Wl,-rpath "target/release" \
        -o "$OUTPUT_DIR/$base_name-march-native" \
        >> /dev/stderr

    hyperfine \
        --shell=none \
        --warmup 3 \
        --export-markdown "$OUTPUT_DIR/$base_name.md" \
        "$CAIRO_RUN --available-gas 18446744073709551615 -s $base_path.cairo" \
        "echo '[null, 18446744073709551615]' | $JIT_CLI $base_path.cairo $base_name::$base_name::main --inputs -" \
        "$OUTPUT_DIR/$base_name" \
        "$OUTPUT_DIR/$base_name-march-native" \
        >> /dev/stderr
}


if [ $# -eq 0 ]
then
    echo "Benchmarking ${#CAIRO_SRCS[@]} programs."
    echo

    count=1
    for program in "${CAIRO_SRCS[@]}"
    do
        echo "[$count/${#CAIRO_SRCS[@]}] Benchmarking program at $program."
        run_bench "$program"

        count=$((count + 1))
    done
else
    echo "Benchmarking ${$#} programs."

    count=1
    for program in "$@"
    do
        echo "[$count/${#CAIRO_SRCS[@]}] Benchmarking program at $program."
        run_bench "$program"

        count=$((count + 1))
    done
fi
