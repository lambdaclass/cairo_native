# cairo-native-fuzz

A crate for fuzzing cairo-native with AFL (American Fuzzy Lop).

## Dependencies

- cargo-afl: `cargo install cargo-afl`.

## Fuzzing the Corelib

Build the fuzzing target:
```bash
cargo afl build --bin fuzz-corelib
```

Generate the corpus. It generates an input example for each function in the program.
```bash
cargo run --bin gen-corpus -- \
    ../test_data_artifacts/programs/corelib.sierra.json \
    corpus/
```

Run the fuzzer.
```bash
cargo afl fuzz -i corpus -o output -- ../target/debug/fuzz-corelib
```

To reproduce a crash, we build with AFL_NO_CFG_FUZZING=1 to enable useful debug prints.

```bash
AFL_NO_CFG_FUZZING=1 cargo afl build --bin fuzz-corelib
 ../target/debug/fuzz-corelib < output/default/crashes/*
```

## Known Crashes

- SIGSEGV on Corelib's core::poseidon::_poseidon_hash_span_inner:
  ```bash
  xxd -r crashes/corelib-poseidon.xxd |
  ../target/debug/fuzz-corelib
  ```
