#!/usr/bin/env bash

# config vars
cairo_run=$S2M_BENCH_CAIRO_RUNNER
sierra2mlir_cli=./target/release/cli
llvm_dir=$MLIR_SYS_160_PREFIX
bench_results=./bench-results

RED="\e[31m"
GREEN="\e[32m"
ENDCOLOR="\e[0m"

mkdir -p $bench_results

# bench programs
for sierra_program in ./sierra2mlir/benches/programs/*.sierra; do
	echo '--------------------'
	echo -e "${RED}BENCHMARKING FILE${ENDCOLOR}: ${GREEN}$sierra_program${ENDCOLOR}"
	echo '>>>>>>>>>>>>>>>>>>>>'
	program="$(basename "$sierra_program" sierra)"
	filename=$(basename -- "$sierra_program")
	filename="${filename%.*}"
	$sierra2mlir_cli compile "$sierra_program" -m --available-gas 9000000000 -o $bench_results/"$filename".mlir
	"$llvm_dir"/bin/mlir-translate --mlir-to-llvmir $bench_results/"$filename".mlir -o $bench_results/"$filename".ll
	"$llvm_dir"/bin/clang $bench_results/"$filename".ll -Wno-override-module -L"$llvm_dir"/lib -L"./target/release/" \
		-lsierra2mlir_utils -lmlir_c_runner_utils -Wl,-rpath "$llvm_dir"/lib \
		-Wl,-rpath "./target/release/" -o $bench_results/"$filename".bin
	hyperfine -N -i --warmup 3 \
			"$cairo_run --available-gas 9000000000 sierra2mlir/benches/programs/${program}cairo" \
			"$sierra2mlir_cli run --available-gas 9000000000 $sierra_program -m -f main" \
			"$bench_results/$filename".bin
	echo '<<<<<<<<<<<<<<<<<<<'
done

# example programs
for sierra_program in ./examples/*.sierra; do
	echo '--------------------'
	echo -e "${RED}BENCHMARKING FILE${ENDCOLOR}: ${GREEN}$sierra_program${ENDCOLOR}"
	echo '>>>>>>>>>>>>>>>>>>>>'
	program="$(basename "$sierra_program" sierra)"
	filename=$(basename -- "$sierra_program")
	filename="${filename%.*}"
	$sierra2mlir_cli compile "$sierra_program" -m --available-gas 9000000000 -o $bench_results/"$filename".mlir
	"$llvm_dir"/bin/mlir-translate --mlir-to-llvmir $bench_results/"$filename".mlir -o $bench_results/"$filename".ll
	"$llvm_dir"/bin/clang $bench_results/"$filename".ll -Wno-override-module -L"$llvm_dir"/lib -L"./target/release/" \
		-lsierra2mlir_utils -lmlir_c_runner_utils -Wl,-rpath "$llvm_dir"/lib \
		-Wl,-rpath "./target/release/" -o $bench_results/"$filename".bin
	hyperfine -N -i --warmup 3 \
			"$cairo_run --available-gas 9000000000 examples/${program}cairo" \
			"$sierra2mlir_cli run --available-gas 9000000000 $sierra_program -m -f main" \
			"$bench_results/$filename".bin
	echo '<<<<<<<<<<<<<<<<<<<'
done
