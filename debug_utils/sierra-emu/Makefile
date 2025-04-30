usage:
	@echo "Usage:"
	@echo "    corelib:	     Installs the necesarry dependencies."
	@echo "    build:        Builds the sierra emu."
	@echo "    check:        Checks format and lints."
	@echo "    test:         Runs all tests."
	@echo "    clean:        Cleans the built artifacts."

corelib: ../../corelib
	ln -s ../../corelib corelib

build:
	cargo build --release --all-features
.PHONY: build

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

test: corelib
	cargo test --all-features
.PHONY: test

clean:
	rm corelib
	cargo clean
.PHONY: clean
