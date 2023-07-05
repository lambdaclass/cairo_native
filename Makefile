.PHONY: book build coverage check clean clean-all test bench

#
# Environment detection.
#

ifeq ($(MLIR_SYS_160_PREFIX),)
  $(error Could not find a suitable LLVM 16 toolchain)
endif

build:
	cargo build --release --all-features

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --profile optimized-dev --all-targets --all-features

coverage:
	cargo llvm-cov --profile optimized-dev --all-features --workspace --lcov --output-path lcov.info

book:
	mdbook serve docs

bench: build
	./scripts/bench-hyperfine.sh

bench-ci:
	cargo criterion --all-features

clean: clean-examples clean-tests clean-bench

clean-all: clean
	-cargo clean
