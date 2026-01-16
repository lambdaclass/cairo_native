# cairo-native-fuzz

A crate for fuzzing cairo-native with AFL (American Fuzzy Lop).

## Dependencies

- cargo-afl: `cargo install cargo-afl`.

## Fuzzing Programs

Build the fuzzing target:
```bash
cargo afl build --bin fuzz-program
```

Generate the corpus. It generates an input example for each function in the program.
```bash
cargo run --bin gen-corpus -- program \
    ../test_data_artifacts/programs/corelib.sierra.json \
    corpus/
```

Run the fuzzer.
```bash
cargo afl fuzz -i corpus -o output -- ../target/debug/fuzz-program
```

To reproduce a crash, we build with AFL_NO_CFG_FUZZING=1 to enable useful debug prints.

```bash
AFL_NO_CFG_FUZZING=1 cargo afl build --bin fuzz-program
 ../target/debug/fuzz-program < output/default/crashes/*
```

## Fuzzing Contracts

Build the fuzzing target:
```bash
cargo afl build --bin fuzz-contract
```

Generate the corpus. It generates an input example for each entrypoint in the contract.
```bash
cargo run --bin gen-corpus -- contract \
    ../test_data_artifacts/contracts/cairo_vm/fib.contract.json \
    corpus/
```

Run the fuzzer.
```bash
cargo afl fuzz -i corpus -o output -- ../target/debug/fuzz-contract
```

To reproduce a crash:
```bash
AFL_NO_CFG_FUZZING=1 cargo afl build --bin fuzz-contract
 ../target/debug/fuzz-contract < output/default/crashes/*
```

## Known Crashes

- SIGSEGV on Corelib's core::poseidon::_poseidon_hash_span_inner:
  ```bash
  xxd -r crashes/corelib-poseidon.xxd |
  ../target/debug/fuzz-program
  ```
