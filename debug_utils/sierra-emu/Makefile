.PHONY: usage deps build check needs-cairo2 deps-macos build-cairo-2-compiler-macos decompress-cairo install-scarb clean

UNAME := $(shell uname)

CAIRO_2_VERSION= 2.12.0-dev.1
SCARB_VERSION = 2.11.2

needs-cairo2:
ifeq ($(wildcard ./cairo2/.),)
	$(error You are missing the Starknet Cairo 1 compiler, please run 'make deps' to install the necessary dependencies.)
endif
	./scripts/check-corelib-version.sh $(CAIRO_2_VERSION)

usage:
	@echo "Usage:"
	@echo "    deps:            Installs the necesarry dependencies."
	@echo "    build:           Builds the cairo-native library and binaries."
	@echo "    check:           Checks format and lints."
	@echo "    test:            Runs all tests."
	@echo "    test-no-corelib: Runs all tests except for the ones from the cairo's corelib."
	@echo "    clean:           Cleans the built artifacts."

build:
	cargo build --release --all-features

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

test: needs-cairo2
	cargo test --all-features

test-no-corelib: needs-cairo2
	cargo test --all-features -- --skip test_corelib

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

install-scarb:
	curl --proto '=https' --tlsv1.2 -sSf https://docs.swmansion.com/scarb/install.sh| sh -s -- --no-modify-path --version $(SCARB_VERSION)

install-scarb-macos:
	curl --proto '=https' --tlsv1.2 -sSf https://docs.swmansion.com/scarb/install.sh| sh -s -- --version $(SCARB_VERSION)
