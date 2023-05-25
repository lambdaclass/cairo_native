#!/bin/bash

for i in $(ls -1 examples/*.sierra); do 
	echo "--------------------"
	echo file: $i;
	echo "    cairo-run"
	time /Users/igaray/src/starkware/cairo/target/release/cairo-run --available-gas 9000000000 examples/$(basename $i sierra)cairo
	echo "    sierra2mlir"
	filename=$(basename -- "$i")
	filename="${filename%.*}"
	NAME=$filename
	time ./target/debug/cli run --available-gas 9000000000 $i --function $NAME::$NAME::main
done

PATH=/Users/igaray/src/starkware/cairo/target/release/:$PATH cairo-compile --replace-ids examples/fibonacci.cairo examples/fibonacci.sierra
RUST_LOG=debug ./target/release/cli compile --optimize examples/fibonacci.sierra --main-print --available-gas 900000000 --output fibonacci.mlir
/opt/homebrew/opt/llvm/bin/mlir-translate  --mlir-to-llvmir fibonacci.mlir -o fibonacci.ll
$MLIR_SYS_160_PREFIX/bin/clang -O3 fibonacci.ll -v -o fibonacci
time ./fibonacci
time /Users/igaray/src/starkware/cairo/target/release/cairo-run --available-gas 9000000000 examples/fibonacci.cairo

PATH=/Users/igaray/src/starkware/cairo/target/release/:$PATH cairo-compile --replace-ids examples/factorial_multirun.cairo examples/factorial_multirun.sierra
RUST_LOG=debug ./target/release/cli compile --optimize examples/factorial_multirun.sierra --main-print --available-gas 900000000 --output factorial_multirun.mlir
/opt/homebrew/opt/llvm/bin/mlir-translate  --mlir-to-llvmir factorial_multirun.mlir -o factorial_multirun.ll
$MLIR_SYS_160_PREFIX/bin/clang -O3 factorial_multirun.ll -v -L/opt/homebrew/opt/llvm/lib/ -lmlir_c_runner_utils -o factorial_multirun
time ./factorial_multirun
time /Users/igaray/src/starkware/cairo/target/release/cairo-run --available-gas 9000000000 examples/factorial_multirun.cairo

PATH=/Users/igaray/src/starkware/cairo/target/release/:$PATH cairo-compile --replace-ids examples/fibonacci_1000_multirun.cairo examples/fibonacci_1000_multirun.sierra
RUST_LOG=debug ./target/release/cli compile --optimize examples/fibonacci_1000_multirun.sierra --main-print --available-gas 900000000 --output fibonacci_1000_multirun.mlir
/opt/homebrew/opt/llvm/bin/mlir-translate  --mlir-to-llvmir fibonacci_1000_multirun.mlir -o fibonacci_1000_multirun.ll
$MLIR_SYS_160_PREFIX/bin/clang -O3 fibonacci_1000_multirun.ll -v -L/opt/homebrew/opt/llvm/lib/ -lmlir_c_runner_utils -o fibonacci_1000_multirun
time ./fibonacci_1000_multirun
time /Users/igaray/src/starkware/cairo/target/release/cairo-run --available-gas 9000000000 examples/fibonacci_1000_multirun.cairo

