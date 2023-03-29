.PHONY: build check clean build-dialect check-mlir build-examples compile-example compile-example-optimized

build: check-mlir
	cargo build --release

check:
	cargo clippy --all

clean:
	cargo clean

test: check-mlir
	cargo test --all

compile-example: check-mlir
	RUST_LOG="debug" cargo r -- compile examples/simple.sierra

compile-example-optimized: check-mlir
	RUST_LOG="debug" cargo r -- compile examples/simple.sierra --optimize

build-examples: check-mlir
	cargo r -- compile examples/print_test.sierra -o examples/print_test.mlir -m
	cargo r -- compile examples/simple.sierra -o examples/simple.mlir
	cargo r -- compile examples/simple.sierra -o examples/simple-optimized.mlir --optimize
	cargo r -- compile examples/simple_enum.sierra -o examples/simple_enum.mlir
	cargo r -- compile examples/felt_is_zero.sierra -o examples/felt_is_zero.mlir
	cargo r -- compile examples/fib.sierra -o examples/fib.mlir
	$(MLIR_SYS_160_PREFIX)/bin/mlir-translate --mlir-to-llvmir examples/simple.mlir -o examples/simple.ll
	$(MLIR_SYS_160_PREFIX)/bin/mlir-translate --mlir-to-llvmir examples/simple-optimized.mlir -o examples/simple-optimized.ll
	$(MLIR_SYS_160_PREFIX)/bin/mlir-translate --mlir-to-llvmir examples/print_test.mlir -o examples/print_test.ll
	$(MLIR_SYS_160_PREFIX)/bin/mlir-translate --mlir-to-llvmir examples/simple_enum.mlir -o examples/simple_enum.ll
	$(MLIR_SYS_160_PREFIX)/bin/mlir-translate --mlir-to-llvmir examples/felt_is_zero.mlir -o examples/felt_is_zero.ll
	$(MLIR_SYS_160_PREFIX)/bin/mlir-translate --mlir-to-llvmir examples/fib.mlir -o examples/fib.ll

book:
	mdbook serve docs

# Experimental, just playing around.
build-dialect: check-mlir
	$(MLIR_SYS_160_PREFIX)/bin/mlir-tblgen --gen-dialect-decls dialect/SierraBase.td -I $(MLIR_SYS_160_PREFIX)/include/ -I ./dialect/ > dialect/SierraBase.h
	$(MLIR_SYS_160_PREFIX)/bin/mlir-tblgen --gen-dialect-defs dialect/SierraBase.td -I $(MLIR_SYS_160_PREFIX)/include/ -I ./dialect/ > dialect/SierraBase.cpp
	$(MLIR_SYS_160_PREFIX)/bin/mlir-tblgen --gen-op-decls dialect/SierraOps.td -I $(MLIR_SYS_160_PREFIX)/include/ -I ./dialect/ > dialect/SierraOps.h
	$(MLIR_SYS_160_PREFIX)/bin/mlir-tblgen --gen-op-defs dialect/SierraOps.td -I $(MLIR_SYS_160_PREFIX)/include/ -I ./dialect/ > dialect/SierraOps.cpp

check-mlir:
ifndef MLIR_SYS_160_PREFIX
	$(error MLIR_SYS_160_PREFIX needs to be set to the path where LLVM with MLIR is)
endif

# requires ninja
install-llvm:
	mkdir -p llvm/dist
	mkdir -p llvm/source
	wget -P llvm/source/ https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.0/llvm-project-16.0.0.src.tar.xz
	tar -xf llvm/source/llvm-project-16.0.0.src.tar.xz -C llvm/source/
	cd llvm/source/llvm-project-16.0.0.src && \
		mkdir -p build && \
		cd build && \
		cmake -G Ninja ../llvm \
			-DLLVM_ENABLE_PROJECTS=mlir \
			-DLLVM_BUILD_EXAMPLES=ON \
			-DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX;AMDGPU" \
			-DCMAKE_BUILD_TYPE=RelWithDebInfo \
			-DLLVM_ENABLE_ASSERTIONS=ON \
			-DLLVM_INSTALL_UTILS=ON \
			-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
			-DCMAKE_INSTALL_PREFIX="../../../dist/" && \
		cmake --build . && \
		cmake --install . && \
		echo "Please create export a variable named MLIR_SYS_160_PREFIX pointing to the absolute path under llvm/dist"
