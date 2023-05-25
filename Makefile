.PHONY: book build coverage check clean clean-all compile-mlir compile-mlir-opt sierra test

#
# Environment detection.
#

LLVM_PREFIX := $(shell scripts/find-llvm.sh)
ifeq ($(LLVM_PREFIX),)
  $(error Could not find a suitable LLVM 16 toolchain)
endif

#
# Source sets.
#

CAIRO_SOURCES := $(wildcard examples/*.cairo)
CAIRO_TARGETS := $(patsubst %.cairo,%.sierra,$(CAIRO_SOURCES))

BENCH_SOURCES := $(wildcard sierra2mlir/benches/programs/*.cairo)
BENCH_TARGETS := $(patsubst %.cairo,%.sierra,$(BENCH_SOURCES))

MLIR_TARGETS     := $(patsubst %.cairo,%.mlir,$(CAIRO_SOURCES))
MLIR_OPT_TARGETS := $(patsubst %.cairo,%.opt.mlir,$(CAIRO_SOURCES))

LLVM_TARGETS     := $(patsubst %.cairo,%.ll,$(CAIRO_SOURCES))
LLVM_OPT_TARGETS := $(patsubst %.cairo,%.opt.ll,$(CAIRO_SOURCES))

COMPARISON_TEST_SOURCES := $(wildcard sierra2mlir/tests/comparison/**/*.cairo)
COMPARISON_TEST_TARGETS := $(patsubst %.cairo,%.sierra,$(COMPARISON_TEST_SOURCES))

#
# Build rules.
#

%.sierra: %.cairo
	cairo-compile --replace-ids $< $@

%.mlir: %.sierra build
	./target/release/cli compile --available-gas 1000000 -o $@ $<

%.opt.mlir: %.sierra build
	./target/release/cli compile  --available-gas 1000000 --optimize -o $@ $<

%.ll: %.mlir
	$(LLVM_PREFIX)/bin/mlir-translate --mlir-to-llvmir -o $@ $<

build:
	cargo build --release

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings

test:
	cargo test --all-targets

coverage:
	cargo llvm-cov --profile "ci-coverage" --all-features --workspace --lcov --output-path lcov.info

book:
	mdbook serve docs

# Compile the cairo sources using `cairo-compile` (must be available on $PATH).
sierra: $(CAIRO_TARGETS) $(BENCH_TARGETS)

# Compile the sierra programs to MLIR using this project.
compile-mlir: $(MLIR_TARGETS)

# Compile the sierra programs to MLIR using this project.
compile-mlir-opt: $(MLIR_OPT_TARGETS)

# Compile the MLIR to llvm ir using mlir-translate
compile-ll: $(LLVM_TARGETS)

# Compile the optimised MLIR to llvm ir using mlir-translate
compile-ll-opt: $(LLVM_OPT_TARGETS)

bench:
	$(shell scripts/comparison.sh)
	cargo bench

clean-examples:
	-rm -rf examples/*.ll examples/*.mlir examples/*.sierra

clean-tests:
	-rm -rf sierra2mlir/benches/programs/*.sierra
	-rm -rf sierra2mlir/tests/comparison/*.sierra
	-rm -rf sierra2mlir/tests/comparison/**/*.sierra
	-rm -rf sierra2mlir/tests/comparison/out/*.ll
	-rm -rf sierra2mlir/tests/comparison/out/*.mlir
	-rm -rf sierra2mlir/tests/comparison/out/*.out

clean: clean-examples clean-tests

clean-all: clean
	-cargo clean
