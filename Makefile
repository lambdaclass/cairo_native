.PHONY: usage build book build-dev build-native coverage check test bench bench-ci doc doc-open install clean

#
# Environment detection.
#

ifeq ($(MLIR_SYS_160_PREFIX),)
  $(error Could not find a suitable LLVM 16 toolchain)
endif

# If corelibs is not present, fetch it.
ifeq ($(wildcard ./corelib/.),)
  $(shell ./scripts/fetch-corelibs.sh)
endif

usage:
	@echo "Usage:"
	@echo "    build:        Builds the cairo-native library and binaries."
	@echo "    build-native: Builds cairo-native with the target-cpu=native rust flag."
	@echo "    build-dev:    Builds cairo-native under a development-optimized profile."
	@echo "    check:        Checks format and lints."
	@echo "    test:         Runs standard tests."
	@echo "    proptest:     Runs property tests."
	@echo "    coverage:     Computes test coverage."
	@echo "    doc:          Builds documentation."
	@echo "    doc-open:     Opens documentation."
	@echo "    book:         Builds and serves the documentation book."
	@echo "    bench:        Runs the hyperfine benchmark script."
	@echo "    bench-ci:     Runs the criterion benchmarks for CI."
	@echo "    install:      Invokes cargo to install cairo-native."
	@echo "    clean:        Cleans the built artifacts."

build:
	cargo build --release --all-features

build-native:
	RUSTFLAGS="-C target-cpu=native" cargo build --release --all-features

build-dev:
	cargo build --profile optimized-dev --all-targets --all-features

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --profile optimized-dev --all-targets --all-features

proptest:
	cargo test --profile optimized-dev --all-targets --all-features proptest

coverage:
	cargo llvm-cov --verbose --profile optimized-dev --all-features --workspace --lcov --output-path lcov.info

doc:
	cargo doc --all-features --no-deps --workspace

doc-open:
	cargo doc --all-features --no-deps --workspace --open

book:
	mdbook serve docs

bench: build
	./scripts/bench-hyperfine.sh

bench-ci:
	cargo criterion --all-features

install:
	RUSTFLAGS="-C target-cpu=native" cargo install --all-features --locked --path .

clean:
	cargo clean
