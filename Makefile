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
	RUST_LOG="debug" cargo r -- -i examples/simple.sierra compile

compile-example-optimized: check-mlir
	RUST_LOG="debug" cargo r -- --optimize -i examples/simple.sierra compile

build-examples: check-mlir
	cargo r -- -i examples/simple.sierra compile -o examples/simple.mlir
	cargo r -- --optimize -i examples/simple.sierra compile -o examples/simple-optimized.mlir
	$(MLIR_SYS_160_PREFIX)/bin/mlir-translate --mlir-to-llvmir examples/simple.mlir -o examples/simple.ll
	$(MLIR_SYS_160_PREFIX)/bin/mlir-translate --mlir-to-llvmir examples/simple-optimized.mlir -o examples/simple-optimized.ll

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
