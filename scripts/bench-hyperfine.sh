#!/usr/bin/env bash

# Configuration.
ROOT_DIR="$(dirname "$(dirname "${0%/*}")")"
MLIR_DIR="$MLIR_SYS_170_PREFIX"

CAIRO_SRCS=$(find \
    "$ROOT_DIR/programs/benches" \
    -type f -name "*.cairo")
IFS=$'\n' read -rd '' -a CAIRO_SRCS <<<"$CAIRO_SRCS"

CAIRO_RUN="$ROOT_DIR/cairo2/bin/cairo-run"
COMPILER_CLI="$ROOT_DIR/target/release/cairo-native-dump"
JIT_CLI="$ROOT_DIR/target/release/cairo-native-run"
OUTPUT_DIR="$ROOT_DIR/target/bench-outputs"

bold=$(tput bold)
normal=$(tput sgr0)

# Initial setup.
if [[ ! -e "$COMPILER_CLI" ]]
then
    echo "${bold}Compiler CLI not found. Is the project built in release mode?${normal}"
    exit 1
fi
if [[ -z "$MLIR_DIR" ]]
then
    echo "${bold}MLIR_DIR is empty. Did you forget to set MLIR_SYS_160_PREFIX?${normal}"
    exit 1
fi

echo "This benchmarks compares the Starknet cairo-run command, which uses the Rust cairo-vm internally against the"
echo "cairo native compiler, using both the JIT engine and a natively compiled binary"
echo

set -e
mkdir -p "$OUTPUT_DIR"

# Benchmarking code.
run_bench() {
    base_path="${1%.cairo}"
    base_name=$(basename $base_path)

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
        --finalize-memref-to-llvm \
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
        --warmup 3 \
        --export-markdown "$OUTPUT_DIR/$base_name.md" \
        --export-json "$OUTPUT_DIR/$base_name.json" \
        -n "Cairo-vm (Rust, Cairo 1)" "$CAIRO_RUN --available-gas 18446744073709551615 -s $base_path.cairo" \
        -n "cairo-native (embedded AOT)" "$JIT_CLI --mode=aot $base_path.cairo $base_name::$base_name::main" \
        -n "cairo-native (embedded JIT using LLVM's ORC Engine)" "$JIT_CLI --mode=jit $base_path.cairo $base_name::$base_name::main" \
        -n "cairo-native (standalone AOT)" "$OUTPUT_DIR/$base_name" \
        -n "cairo-native (standalone AOT with -march=native)" "$OUTPUT_DIR/$base_name-march-native" \
        >> /dev/stderr
}

echo "Rust cairo-run version: $($CAIRO_RUN --version)"

if [ $# -eq 0 ]
then
    echo "${bold}Benchmarking ${#CAIRO_SRCS[@]} programs.${normal}"

    count=1
    for program in "${CAIRO_SRCS[@]}"
    do
        echo "${bold}[$count/${#CAIRO_SRCS[@]}] Benchmarking program at $program.${normal}"
        run_bench "$program"

        count=$((count + 1))
    done
else
    echo "${bold}Benchmarking $# programs.${normal}"

    count=1
    for program in "$@"
    do
        echo "${bold}[$count/${#CAIRO_SRCS[@]}] Benchmarking program at $program.${normal}"
        run_bench "$program"

        count=$((count + 1))
    done
fi
