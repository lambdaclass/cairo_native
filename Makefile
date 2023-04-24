.PHONY: book build check clean clean-all compile-mlir compile-mlir-opt sierra test


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
	./target/release/cli compile -o $@ $<

%.opt.mlir: %.sierra build
	./target/release/cli compile --optimize -o $@ $<

%.ll: %.mlir
	$(LLVM_PREFIX)/bin/mlir-translate --mlir-to-llvmir -o $@ $<

build:
	cargo build --release

check:
	cargo clippy --all-targets -- -D warnings

test:
	cargo test --all-targets

book:
	mdbook serve docs


# Compile the cairo sources using `cairo-compile` (must be available on $PATH).
sierra: $(CAIRO_TARGETS)

# Compile the sierra programs to MLIR using this project.
compile-mlir: $(MLIR_TARGETS)

# Compile the sierra programs to MLIR using this project.
compile-mlir-opt: $(MLIR_OPT_TARGETS)

# Compile the MLIR to llvm ir using mlir-translate
compile-ll: $(LLVM_TARGETS)

# Compile the optimised MLIR to llvm ir using mlir-translate
compile-ll-opt: $(LLVM_OPT_TARGETS)


clean-examples:
	-rm -rf examples/*.ll examples/*.mlir examples/*.sierra

clean-tests:
	-rm -rf sierra2mlir/tests/comparison/*.sierra
	-rm -rf sierra2mlir/tests/comparison/**/*.sierra
	-rm -rf sierra2mlir/tests/comparison/out/*.ll
	-rm -rf sierra2mlir/tests/comparison/out/*.mlir

clean: clean-examples clean-tests

clean-all: clean
	-cargo clean
