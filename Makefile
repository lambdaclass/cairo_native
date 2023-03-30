.PHONY: book build check clean clean-all compile-mlir compile-mlir-opt sierra test


#
# Environment detection.
#

LLVM_PREFIX := $(shell scripts/find-llvm.sh)
ifeq ($(LLVM_PREFIX),)
  $(error Could not find a suitable LLVM 16 toolchain)
endif

ifeq ($(shell which cairo-compile),)
  $(error Could not find `cairo-compile`)
endif


#
# Source sets.
#

CAIRO_SOURCES := $(wildcard examples/*.cairo)
CAIRO_TARGETS := $(patsubst %.cairo,%.sierra,$(CAIRO_SOURCES))

MLIR_TARGETS     := $(patsubst %.cairo,%.mlir,$(CAIRO_SOURCES))
MLIR_OPT_TARGETS := $(patsubst %.cairo,%.opt.mlir,$(CAIRO_SOURCES))


#
# Build rules.
#

%.sierra: %.cairo
	cairo-compile --replace-ids $< $@

%.mlir: build %.sierra
	./target/release/cli -o $@ $<

%.opt.mlir: build %.sierra
	./target/release/cli --optimize -o $@ $<


build:
	cargo build --release

check:
	cargo clippy --all-targets

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


clean:
	-rm -rf examples/*.ll examples/*.mlir examples/*.sierra

clean-all: clean
	-cargo clean
