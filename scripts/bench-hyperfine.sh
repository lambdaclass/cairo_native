#!/usr/bin/env bash

# Configuration.
ROOT_DIR=$(dirname $(dirname ${0%/*}))
MLIR_DIR=$("$ROOT_DIR/scripts/find-llvm.sh")

CAIRO_SRCS=$(find \
    "$ROOT_DIR/sierra2mlir/benches/programs" \
    "$ROOT_DIR/examples" \
-type f -name "*.cairo")
IFS=$'\n' read -rd '' -a CAIRO_SRCS <<<"$CAIRO_SRCS"

COMPILER_CLI="$ROOT_DIR/target/release/cli"
OUTPUT_DIR="$ROOT_DIR/target/bench-outputs"


# Initial setup.
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
        compile \
        "$base_path.cairo" \
        --main-print \
        --available-gas 100000000000 \
        --output "$OUTPUT_DIR/$base_name.mlir" \
        >> /dev/stderr

    "$MLIR_DIR/bin/mlir-translate" \
        --mlir-to-llvmir \
        "$OUTPUT_DIR/$base_name.mlir" \
        -o "$OUTPUT_DIR/$base_name.ll" \
        >> /dev/stderr

    "$MLIR_DIR/bin/clang" \
        "$OUTPUT_DIR/$base_name.ll" \
        -o "$OUTPUT_DIR/$base_name" \
        >> /dev/stderr

    hyperfine \
        --shell=none \
        --ignore-failure \
        --export-markdown "$OUTPUT_DIR/$base_name.md" \
        --warmup 3 \
        "cairo-run --available-gas 100000000000 $base_path.cairo" \
        "$COMPILER_CLI run --available-gas 100000000000 $base_path.cairo -m -f main" \
        "$OUTPUT_DIR/$base_name" \
        >> /dev/stderr
}


echo "Benchmarking ${#CAIRO_SRCS[@]} programs."
echo

count=1
for program in "${CAIRO_SRCS[@]}"
do
    echo "[$count/${#CAIRO_SRCS[@]}] Benchmarking program at $program."
    run_bench "$program"

    count=$((count + 1))
done
