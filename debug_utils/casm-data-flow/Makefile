usage:
	@echo "Usage:"
	@echo "    build: Builds casm-data-flow."
	@echo "    check: Checks format and lints."
	@echo "    clean: Cleans the built artifacts."
.PHONY: usage

build:
	cargo build --release --all-features
.PHONY: build

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
.PHONY: check

clean:
	cargo clean
.PHONY: clean
