.PHONY: build book build-dev coverage check test bench doc doc-open clean

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

build:
	cargo build --release --all-features

build-dev:
	cargo build --profile optimized-dev --all-targets --all-features

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --profile optimized-dev --all-targets --all-features

coverage:
	cargo llvm-cov --verbose --profile release --all-features --workspace --lcov --output-path lcov.info

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
	cargo install --all-features --locked --path .

clean:
	cargo clean
