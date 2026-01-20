usage:
	@echo "Usage:"
	@echo "    corelib:         Install the corelib."
	@echo "    build:           Builds the sierra emu."
	@echo "    check:           Checks format and lints."
	@echo "    test:            Runs all tests."
	@echo "    test-no-corelib: Runs all tests except for the ones from the cairo's corelib."
	@echo "    clean:           Cleans the built artifacts."

build:
	cargo build --release --all-features
.PHONY: build

check:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
.PHONY: check

test:
	python3 ./scripts/compile_test_data.py --mode sierra-emu
	cargo test --all-features
.PHONY: test

test-no-corelib:
	python3 ./scripts/compile_test_data.py --mode sierra-emu
	cargo test --all-features -- --skip test_corelib
.PHONY: test-no-corelib

clean:
	rm corelib
	cargo clean
.PHONY: clean
