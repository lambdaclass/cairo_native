# ⚡ Cairo Sierra Emulator ⚡

An Cairo emulator directly using the Cairo's intermediate representation "Sierra" instead of CASM. An useful usecase is to aid in debugging Cairo Native.

## Dependencies

First, make sure to have all the dependencies from Cairo Native setup.

Then, you can use the corelib recipe to make a symlink in the current directory.
```bash
make corelib
```

## Running the Program

To use the sierra emulator binary, we must first compile the target cairo program.

```bash
../../cairo2/bin/cairo-compile -rs ./programs/fibonacci.cairo > ./programs/fibonacci.sierra
```

Then, we can generate the ejecution trace with the sierra emulator:

```bash
cargo run -- ./programs/fibonacci.sierra fibonacci::fibonacci::main \
    --available-gas 100000 --output ./programs/fibonacci.trace.json
```

The program trace will be generated to `./programs/fibonacci.trace.json`.

## Using the API

The sierra emulator can also be used as a library.

- See [examples/program.rs](examples/program.rs) for an example on how to execute a cairo program
- See [examples/contract.rs](examples/contract.rs) for an example on how to execute a cairo contract
