#!/usr/bin/env bash

# config vars
sierra2mlir_cli=./target/release/cli
llvm_dir=$MLIR_SYS_160_PREFIX
bench_results=./bench-results

RED="\e[31m"
GREEN="\e[32m"
ENDCOLOR="\e[0m"

mkdir -p $bench_results

# bench programs
for program_source in ./sierra2mlir/benches/programs/*.cairo; do
	echo '--------------------'
	echo -e "${RED}BENCHMARKING FILE${ENDCOLOR}: ${GREEN}$program_source${ENDCOLOR}"
	echo '>>>>>>>>>>>>>>>>>>>>'
	program="$(basename "$program_source" cairo)"
	filename=$(basename -- "$program_source")
	filename="${filename%.*}"
	$sierra2mlir_cli compile "$program_source" -m --available-gas 9000000000 -o $bench_results/"$filename".mlir
	"$llvm_dir"/bin/mlir-translate --mlir-to-llvmir $bench_results/"$filename".mlir -o $bench_results/"$filename".ll
	"$llvm_dir"/bin/clang $bench_results/"$filename".ll -Wno-override-module -L"$llvm_dir"/lib -L"./target/release/" \
		-lsierra2mlir_utils -lmlir_c_runner_utils -Wl,-rpath "$llvm_dir"/lib \
		-Wl,-rpath "./target/release/" -o $bench_results/"$filename".bin
	hyperfine -N -i --warmup 3 \
			"cairo-run --available-gas 9000000000 sierra2mlir/benches/programs/${program}cairo" \
			"$sierra2mlir_cli run --available-gas 9000000000 $program_source -m -f main" \
			"$bench_results/$filename".bin
	echo '<<<<<<<<<<<<<<<<<<<'
done

# example programs
for program_source in ./examples/*.cairo; do
	echo '--------------------'
	echo -e "${RED}BENCHMARKING FILE${ENDCOLOR}: ${GREEN}$program_source${ENDCOLOR}"
	echo '>>>>>>>>>>>>>>>>>>>>'
	program="$(basename "$program_source" cairo)"
	filename=$(basename -- "$program_source")
	filename="${filename%.*}"
	$sierra2mlir_cli compile "$program_source" -m --available-gas 9000000000 -o $bench_results/"$filename".mlir
	"$llvm_dir"/bin/mlir-translate --mlir-to-llvmir $bench_results/"$filename".mlir -o $bench_results/"$filename".ll
	"$llvm_dir"/bin/clang $bench_results/"$filename".ll -Wno-override-module -L"$llvm_dir"/lib -L"./target/release/" \
		-lsierra2mlir_utils -lmlir_c_runner_utils -Wl,-rpath "$llvm_dir"/lib \
		-Wl,-rpath "./target/release/" -o $bench_results/"$filename".bin
	hyperfine -N -i --warmup 3 \
			"cairo-run --available-gas 9000000000 examples/${program}cairo" \
			"$sierra2mlir_cli run --available-gas 9000000000 $program_source -m -f main" \
			"$bench_results/$filename".bin
	echo '<<<<<<<<<<<<<<<<<<<'
done
