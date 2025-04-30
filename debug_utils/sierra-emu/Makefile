usage:
	@echo "Usage:"
	@echo "    corelib:	     Install the corelib."
	@echo "    build:        Builds the sierra emu."
	@echo "    check:        Checks format and lints."
	@echo "    test:         Runs all tests."
	@echo "    clean:        Cleans the built artifacts."

corelib: ../../cairo2/corelib
	ln -s ../../cairo2/corelib corelib

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
