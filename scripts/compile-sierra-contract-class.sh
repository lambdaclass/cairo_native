#!/usr/bin/env bash
set -e

print_usage() {
cat <<EOF 
Expected 3 arguments, received $1.

Usage: $0 <contract_class_path> <opt_lvl> <output_directory>

Compiles a Sierra Contract Class, generates the MLIR and LLVMIR files 
and saves then along with the Sierra and CASM files in <output_directory>/.

EOF
}

if ! [ "$#" -ge "3" ]; then
    print_usage $#
    exit 1
fi

if ! [ -d "cairo2/" ]; then
    echo "cairo2/ directory is not present, please run make deps"
    exit 1
fi

if [[ -z "${LLVM_SYS_191_PREFIX}" || -z "${MLIR_SYS_190_PREFIX}" || -z "${TABLEGEN_190_PREFIX}" ]]; then
    echo "Could not find a suitable LLVM 19 toolchain, please the following env vars pointing to the LLVM 19 dir:
        - LLVM_SYS_191_PREFIX
        - MLIR_SYS_190_PREFIX
        - TABLEGEN_190_PREFIX"
    exit 1
fi

CONTRACT_PATH=$1
OPT_LVL=$2 
DEST_DIR=$3
CONTRACT_CLASS_FILE=$(basename $CONTRACT_PATH)
CLASS_HASH=${CONTRACT_CLASS_FILE%.*.*}
SIERRA_PATH=$CLASS_HASH.sierra
CASM_PATH=$CLASS_HASH.casm

# Extract the sierra from the contract class.
cargo run -p debug_utils --package contract-to-sierra $CONTRACT_PATH > $SIERRA_PATH

# Lower sierra to casm
./cairo2/bin/sierra-compile $SIERRA_PATH $CASM_PATH

echo "Compiling contract class..."
# Set NATIVE_DEBUG_DUMP to generate mlir files.
NATIVE_DEBUG_DUMP=true cargo run --release --package starknet-native-compile -- -O $OPT_LVL $CONTRACT_PATH output

echo "Converting optimized mlir into llvmir unoptimized..."
$LLVM_SYS_191_PREFIX/bin/mlir-translate -mlir-to-llvmir dump.mlir > dump-prepass.ll

echo "Optimizing llvmir..."
$LLVM_SYS_191_PREFIX/bin/opt dump-prepass.ll -passes="default<O$OPT_LVL>" -S -o dump-opt.ll

echo "Saving generated files"
mkdir $DEST_DIR
mv $SIERRA_PATH $CASM_PATH $DEST_DIR 
mv dump.mlir dump-debug-pretty.mlir dump-prepass-debug-pretty.mlir dump-prepass.ll dump-opt.ll dump-debug.mlir dump-prepass.mlir dump-prepass-debug-valid.mlir $DEST_DIR 

echo "Cleaning..."
rm output.json
rm output
