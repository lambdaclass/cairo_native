.PHONY: usage build book build-dev build-native coverage check test bench bench-ci doc doc-open install clean build-alexandria install-scarb

#
# Environment detection.
#

UNAME := $(shell uname)
CAIRO_2_VERSION=2.3.1

check-llvm:
ifndef MLIR_SYS_170_PREFIX
	$(error Could not find a suitable LLVM 17 toolchain (mlir), please set MLIR_SYS_170_PREFIX env pointing to the LLVM 17 dir)
endif
ifndef TABLEGEN_170_PREFIX
	$(error Could not find a suitable LLVM 17 toolchain (tablegen), please set TABLEGEN_170_PREFIX env pointing to the LLVM 17 dir)
endif
	@echo "LLVM is correctly set at $(MLIR_SYS_170_PREFIX)."

needs-cairo2:
ifeq ($(wildcard ./cairo2/.),)
	$(error You are missing the Starknet Cairo 1 compiler, please run 'make deps' to install the necessary dependencies.)
endif
	./scripts/check-corelib-version.sh $(CAIRO_2_VERSION)

usage:
	@echo "Usage:"
	@echo "    deps:		 Installs the necesarry dependencies."
	@echo "    build:        Builds the cairo-native library and binaries."
	@echo "    build-native: Builds cairo-native with the target-cpu=native rust flag."
	@echo "    build-dev:    Builds cairo-native under a development-optimized profile."
	@echo "    check:        Checks format and lints."
	@echo "    test:         Runs all tests."
	@echo "    proptest:     Runs property tests."
	@echo "    coverage:     Runs all tests and computes test coverage."
	@echo "    doc:          Builds documentation."
	@echo "    doc-open:     Builds and opens documentation in browser."
	@echo "    bench:        Runs the hyperfine benchmark script."
	@echo "    bench-ci:     Runs the criterion benchmarks for CI."
	@echo "    install:      Invokes cargo to install cairo-native."
	@echo "    clean:        Cleans the built artifacts."

build: check-llvm
	cargo build --release --all-features

build-native: check-llvm
	RUSTFLAGS="-C target-cpu=native" cargo build --release --all-features

build-dev: check-llvm
	cargo build --profile optimized-dev --all-targets --all-features

check: check-llvm
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

test: check-llvm needs-cairo2 build-alexandria
	cargo test --profile optimized-dev --all-targets --all-features

proptest: check-llvm needs-cairo2
	cargo test --profile optimized-dev --all-targets --all-features proptest

coverage: check-llvm needs-cairo2
	cargo llvm-cov --verbose --profile optimized-dev --all-features --workspace --lcov --output-path lcov.info

doc: check-llvm
	cargo doc --all-features --no-deps --workspace

doc-open: check-llvm
	cargo doc --all-features --no-deps --workspace --open

bench: build needs-cairo2
	./scripts/bench-hyperfine.sh

bench-ci: check-llvm needs-cairo2
	cargo criterion --all-features

install: check-llvm
	RUSTFLAGS="-C target-cpu=native" cargo install --all-features --locked --path .

clean:
	cargo clean

deps:
ifeq ($(UNAME), Linux)
deps: build-cairo-2-compiler install-scarb
endif
ifeq ($(UNAME), Darwin)
deps: build-cairo-2-compiler-macos deps-macos install-scarb-macos
endif
	-rm -rf corelib
	-ln -s cairo2/corelib corelib

deps-macos: build-cairo-2-compiler-macos
	-brew install llvm@17 --quiet
	@echo "You can execute the env-macos.sh script to setup the needed env variables."

cairo-repo-2-dir = cairo2
cairo-repo-2-dir-macos = cairo2-macos

build-cairo-2-compiler-macos: | $(cairo-repo-2-dir-macos)

$(cairo-repo-2-dir-macos): cairo-${CAIRO_2_VERSION}-macos.tar
	$(MAKE) decompress-cairo SOURCE=$< TARGET=cairo2/

build-cairo-2-compiler: | $(cairo-repo-2-dir)

$(cairo-repo-2-dir): cairo-${CAIRO_2_VERSION}.tar
	$(MAKE) decompress-cairo SOURCE=$< TARGET=cairo2/

decompress-cairo:
	rm -rf $(TARGET) \
	&& tar -xzvf $(SOURCE) \
	&& mv cairo/ $(TARGET)

cairo-%-macos.tar:
	curl -L -o "$@" "https://github.com/starkware-libs/cairo/releases/download/v$*/release-aarch64-apple-darwin.tar"

cairo-%.tar:
	curl -L -o "$@" "https://github.com/starkware-libs/cairo/releases/download/v$*/release-x86_64-unknown-linux-musl.tar.gz"

SCARB_VERSION = 2.4.0

install-scarb-macos:
	curl -L -o scarb-$(SCARB_VERSION).tar https://github.com/software-mansion/scarb/releases/download/v2.4.0/scarb-v2.4.0-aarch64-apple-darwin.tar.gz
	tar -xzvf scarb-$(SCARB_VERSION).tar
	mv scarb-v$(SCARB_VERSION)-aarch64-apple-darwin/bin/scarb scarb
	rm -rf scarb-v$(SCARB_VERSION)-aarch64-apple-darwin
	rm -rf scarb-$(SCARB_VERSION).tar

install-scarb:
	curl -L -o scarb-$(SCARB_VERSION).tar https://github.com/software-mansion/scarb/releases/download/v2.4.0/scarb-v2.4.0-x86_64-unknown-linux-musl.tar.gz
	tar -xzvf scarb-$(SCARB_VERSION).tar
	mv scarb-v$(SCARB_VERSION)-x86_64-unknown-linux-musl/bin/scarb scarb
	rm -rf scarb-v$(SCARB_VERSION)-x86_64-unknown-linux-musl
	rm -rf scarb-$(SCARB_VERSION).tar

build-alexandria:
	cd tests/alexandria; ../../scarb build
