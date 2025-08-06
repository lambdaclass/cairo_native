#!/usr/bin/env bash
set -e

print_usage() {
cat <<EOF 
Missing parameters to provide.

Usage: $0 <contract_class_path> <opt_lvl> <output_path>

Compiles a Sierra Contract Class, generates the MLIR and LLVMIR files 
and saves then along with the Sierra and CASM files in <output_path>-files/.

EOF
}

if ! [ "$#" -ge "3" ]; then
    print_usage
    exit 1
fi

if ! [ -d "cairo2/" ]; then
    echo "cairo2/ directory is not present, please run make deps"
    exit 1
fi

CONTRACT_PATH=$1
OPT_LVL=$2 
OUTPUT_PATH=$3
CLASS_HASH=${OUTPUT_PATH%.*}
DEST_DIR=$CLASS_HASH-files
SIERRA_PATH=$CLASS_HASH.sierra
CASM_PATH=$CLASS_HASH.casm

if [[ "$(uname -s)" == "Darwin" ]]; then
    LLVM_PATH="$(brew --prefix llvm@19)"
elif [[ "$(uname -s)" == "Linux" ]]; then
    LLVM_PATH="/usr/lib/llvm-19"
else
    echo "Unsupported OS: $(uname -s)"
fi

# Extract the sierra from the contract class.
cargo run -p contract-utils --bin contract-to-sierra $CONTRACT_PATH >> $SIERRA_PATH

# Lower sierra to casm
./cairo2/bin/sierra-compile $SIERRA_PATH $CASM_PATH

echo "Compiling contract class..."
# Set NATIVE_DEBUG_DUMP to generate mlir files.
NATIVE_DEBUG_DUMP=true cargo run --release --bin starknet-native-compile -- -O $OPT_LVL $CONTRACT_PATH $OUTPUT_PATH

echo "Converting optimized mlir into llvmir unoptimized..."
$LLVM_PATH/bin/mlir-translate -mlir-to-llvmir dump.mlir > dump-prepass.ll

echo "Optimizing llvmir..."
$LLVM_PATH/bin/opt dump-prepass.ll -passes="default<O$OPT_LVL>" -S -o dump-opt.ll

echo "Saving generated files"
mkdir $DEST_DIR
mv $SIERRA_PATH $CASM_PATH $DEST_DIR 
mv dump.mlir dump-debug-pretty.mlir dump-prepass-debug-pretty.mlir dump-prepass.ll dump-opt.ll dump-debug.mlir dump-prepass.mlir dump-prepass-debug-valid.mlir $DEST_DIR 

echo "Cleaning..."
rm $CLASS_HASH.json
rm $OUTPUT_PATH
