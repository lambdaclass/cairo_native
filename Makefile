# Environment detection.

UNAME := $(shell uname)
CAIRO_2_VERSION = 2.8.2
SCARB_VERSION = 2.8.2

# Usage is the default target for newcomers running `make`.
.PHONY: usage
usage: check-llvm needs-cairo2
	@echo "Usage:"
	@echo "    deps:         Installs the necesary dependencies."
	@echo "    build:        Builds the cairo-native library and binaries in release mode."
	@echo "    build-native: Builds cairo-native with the target-cpu=native rust flag."
	@echo "    build-dev:    Builds cairo-native under a development-optimized profile."
	@echo "    runtime:      Builds the runtime library required for AOT compilation."
	@echo "    check:        Checks format and lints."
	@echo "    test:         Runs all tests."
	@echo "    proptest:     Runs property tests."
	@echo "    coverage:     Runs all tests and computes test coverage."
	@echo "    doc:          Builds documentation."
	@echo "    doc-open:     Builds and opens documentation in browser."
	@echo "    bench:        Runs the hyperfine benchmark script."
	@echo "    bench-ci:     Runs the criterion benchmarks for CI."
	@echo "    install:      Invokes cargo to install cairo-native tools."
	@echo "    clean:        Cleans the built artifacts."
	@echo "    stress-test   Runs a command which runs stress tests."
	@echo "    stress-plot   Plots the results of the stress test command."
	@echo "    stress-clean  Clean the cache of AOT compiled code of the stress test command."

.PHONY: check-llvm
check-llvm:
ifndef MLIR_SYS_180_PREFIX
	$(error Could not find a suitable LLVM 18 toolchain (mlir), please set MLIR_SYS_180_PREFIX env pointing to the LLVM 18 dir)
endif
ifndef TABLEGEN_180_PREFIX
	$(error Could not find a suitable LLVM 18 toolchain (tablegen), please set TABLEGEN_180_PREFIX env pointing to the LLVM 18 dir)
endif
	@echo "LLVM is correctly set at $(MLIR_SYS_180_PREFIX)."

.PHONY: needs-cairo2
needs-cairo2:
ifeq ($(wildcard ./cairo2/.),)
	$(error You are missing the Starknet Cairo 1 compiler, please run 'make deps' to install the necessary dependencies.)
endif
	./scripts/check-corelib-version.sh $(CAIRO_2_VERSION)

.PHONY: build
build: check-llvm runtime
	cargo build --release --all-features

.PHONY: build-natives
build-native: check-llvm runtime
	RUSTFLAGS="-C target-cpu=native" cargo build --release --all-features

.PHONY: build-dev
build-dev: check-llvm
	cargo build --profile optimized-dev --all-features

.PHONY: check
check: check-llvm
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

.PHONY: test
test: check-llvm needs-cairo2 build-alexandria runtime-ci
	cargo test --profile ci --all-features

.PHONY: test-cairo
test-cairo: check-llvm needs-cairo2 build-alexandria runtime-ci
	cargo r --profile ci --bin cairo-native-test -- corelib

.PHONY: proptest
proptest: check-llvm needs-cairo2 runtime-ci
	cargo test --profile ci --all-features proptest

.PHONY: test-cli
test-ci: check-llvm needs-cairo2 build-alexandria runtime-ci
	cargo test --profile ci --all-features

.PHONY: proptest-cli
proptest-ci: check-llvm needs-cairo2 runtime-ci
	cargo test --profile ci --all-features proptest

.PHONY: coverage
coverage: check-llvm needs-cairo2 build-alexandria runtime-ci
	cargo llvm-cov --verbose --profile ci --all-features --workspace --lcov --output-path lcov.info
	cargo llvm-cov --verbose --profile ci --all-features --lcov --output-path lcov-test.info run --bin cairo-native-test -- corelib

.PHONY: doc
doc: check-llvm
	cargo doc --all-features --no-deps --workspace

.PHONY: doc-open
doc-open: check-llvm
	cargo doc --all-features --no-deps --workspace --open

.PHONY: bench
bench: build needs-cairo2 runtime
	./scripts/bench-hyperfine.sh

.PHONY: bench-ci
bench-ci: check-llvm needs-cairo2 runtime
	cargo criterion --all-features

.PHONY: stress-test
stress-test: check-llvm
	RUST_LOG=cairo_native_stress=DEBUG cargo run --bin cairo-native-stress 1000000 --output cairo-native-stress-logs.jsonl

.PHONY: stress-plot
stress-plot:
	python3 src/bin/cairo-native-stress/plotter.py cairo-native-stress-logs.jsonl

.PHONY: stress-clean
stress-clean:
	rm -rf .aot-cache

.PHONY: install
install: check-llvm
	RUSTFLAGS="-C target-cpu=native" cargo install --all-features --locked --path .

.PHONY: clean
clean: stress-clean
	cargo clean

.PHONY: deps
deps:
ifeq ($(UNAME), Linux)
deps: build-cairo-2-compiler install-scarb
endif
ifeq ($(UNAME), Darwin)
deps: deps-macos
endif
	-rm -rf corelib
	-ln -s cairo2/corelib corelib

.PHONY: deps-macos
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

.PHONY: decompress-cairo
decompress-cairo:
	rm -rf $(TARGET) \
	&& tar -xzvf $(SOURCE) \
	&& mv cairo/ $(TARGET)

cairo-%-macos.tar:
	curl -L -o "$@" "https://github.com/starkware-libs/cairo/releases/download/v$*/release-aarch64-apple-darwin.tar"

cairo-%.tar:
	curl -L -o "$@" "https://github.com/starkware-libs/cairo/releases/download/v$*/release-x86_64-unknown-linux-musl.tar.gz"

.PHONY: install-scarb
install-scarb:
	curl --proto '=https' --tlsv1.2 -sSf https://docs.swmansion.com/scarb/install.sh| sh -s -- --no-modify-path --version $(SCARB_VERSION)

.PHONY: install-scarb-macos
install-scarb-macos:
	curl --proto '=https' --tlsv1.2 -sSf https://docs.swmansion.com/scarb/install.sh| sh -s -- --version $(SCARB_VERSION)

.PHONY: build-alexandria
build-alexandria:
	cd tests/alexandria; scarb build

.PHONY: runtime
runtime:
	cargo b --release --all-features -p cairo-native-runtime && cp target/release/libcairo_native_runtime.a .

.PHONY: runtime-ci
runtime-ci:
	cargo b --profile ci --all-features -p cairo-native-runtime && cp target/ci/libcairo_native_runtime.a .
