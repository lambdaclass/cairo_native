.PHONY: usage build book build-dev build-native coverage check test bench bench-ci doc doc-open install clean install-scarb install-scarb-macos build-alexandria runtime test-ci proptest-ci

#
# Environment detection.
#

UNAME := $(shell uname)
CAIRO_2_VERSION=2.7.1

check-llvm:
ifndef MLIR_SYS_180_PREFIX
	$(error Could not find a suitable LLVM 18 toolchain (mlir), please set MLIR_SYS_180_PREFIX env pointing to the LLVM 18 dir)
endif
ifndef TABLEGEN_180_PREFIX
	$(error Could not find a suitable LLVM 18 toolchain (tablegen), please set TABLEGEN_180_PREFIX env pointing to the LLVM 18 dir)
endif
	@echo "LLVM is correctly set at $(MLIR_SYS_180_PREFIX)."

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

build: check-llvm runtime
	cargo build --release --features build-cli,with-cheatcode,with-runtime,with-serde

build-native: check-llvm runtime
	RUSTFLAGS="-C target-cpu=native" cargo build --release --features build-cli,with-cheatcode,with-runtime,with-serde

build-dev: check-llvm
	cargo build --profile optimized-dev --features build-cli,with-cheatcode,with-runtime,with-serde

check: check-llvm
	cargo fmt --all -- --check
	cargo clippy --all-targets --features build-cli,with-cheatcode,with-runtime,with-serde -- -D warnings

test: check-llvm needs-cairo2 build-alexandria runtime-ci
	cargo test --profile ci --features build-cli,with-cheatcode,with-runtime,with-serde

test-cairo: check-llvm needs-cairo2 build-alexandria runtime-ci
	cargo r --profile ci --bin cairo-native-test -- corelib

proptest: check-llvm needs-cairo2 runtime-ci
	cargo test --profile ci --features build-cli,with-cheatcode,with-runtime,with-serde proptest

test-ci: check-llvm needs-cairo2 build-alexandria runtime-ci
	cargo test --profile ci --features build-cli,with-cheatcode,with-runtime,with-serde

proptest-ci: check-llvm needs-cairo2 runtime-ci
	cargo test --profile ci --features build-cli,with-cheatcode,with-runtime,with-serde proptest

coverage: check-llvm needs-cairo2 build-alexandria runtime-ci
	cargo llvm-cov --verbose --profile ci --features build-cli,with-cheatcode,with-runtime,with-serde --workspace --lcov --output-path lcov.info
	cargo llvm-cov --verbose --profile ci --features build-cli,with-cheatcode,with-runtime,with-serde --lcov --output-path lcov-test.info run --bin cairo-native-test -- corelib

doc: check-llvm
	cargo doc --features build-cli,with-cheatcode,with-runtime,with-serde --no-deps --workspace

doc-open: check-llvm
	cargo doc --features build-cli,with-cheatcode,with-runtime,with-serde --no-deps --workspace --open

bench: build needs-cairo2 runtime
	./scripts/bench-hyperfine.sh

bench-ci: check-llvm needs-cairo2 runtime
	cargo criterion --features build-cli,with-cheatcode,with-runtime,with-serde

stress-test: check-llvm
	RUST_LOG=cairo_native_stress=DEBUG cargo run --bin cairo-native-stress 1000000 --output cairo-native-stress-logs.jsonl

stress-plot:
	python3 src/bin/cairo-native-stress/plotter.py cairo-native-stress-logs.jsonl

stress-clean:
	rm -rf .aot-cache

install: check-llvm
	RUSTFLAGS="-C target-cpu=native" cargo install --features build-cli,with-cheatcode,with-runtime,with-serde --locked --path .

clean:
	cargo clean

deps:
ifeq ($(UNAME), Linux)
deps: build-cairo-2-compiler install-scarb
endif
ifeq ($(UNAME), Darwin)
deps: deps-macos
endif
	-rm -rf corelib
	-ln -s cairo2/corelib corelib

deps-macos: build-cairo-2-compiler-macos install-scarb-macos
	-brew install llvm@18 --quiet
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

SCARB_VERSION = 2.7.1

install-scarb:
	curl --proto '=https' --tlsv1.2 -sSf https://docs.swmansion.com/scarb/install.sh| sh -s -- --no-modify-path --version $(SCARB_VERSION)

install-scarb-macos:
	curl --proto '=https' --tlsv1.2 -sSf https://docs.swmansion.com/scarb/install.sh| sh -s -- --version $(SCARB_VERSION)

build-alexandria:
	cd tests/alexandria; scarb build

runtime:
	cargo b --release -p cairo-native-runtime && cp target/release/libcairo_native_runtime.a .

runtime-with-trace-dump:
	cargo b --release -p cairo-native-runtime --features=with-trace-dump && cp target/release/libcairo_native_runtime.a .

runtime-ci:
	cargo b --profile ci -p cairo-native-runtime && cp target/ci/libcairo_native_runtime.a .
