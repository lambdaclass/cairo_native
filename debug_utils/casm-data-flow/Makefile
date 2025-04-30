usage:
	@echo "Usage:"
	@echo "    build:        Builds casm-data-flow."
	@echo "    check:        Checks format and lints."
	@echo "    clean:        Cleans the built artifacts."

build:
	cargo build --release --all-features
.PHONY: build

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings

clean:
	cargo clean
.PHONY: clean
