# Casm Data Flow

A data flow analyzer for a Cairo VM. Currently, it only supports starknet contracts.

## Usage

First, we need to compile the contract:

```bash
../../cairo2/bin/starknet-compile -s programs/fibonacci_starknet.cairo > programs/fibonacci_starknet.sierra.json
../../cairo2/bin/starknet-sierra-compile programs/fibonacci_starknet.sierra.json programs/fibonacci_starknet.casm.json
```

First, we need to execute the contract. In this example, we are using [our Cairo VM](https://github.com/lambdaclass/cairo-vm).

```bash
cargo run --example run-contract -- programs/fibonacci_starknet.casm.json programs/fibonacci_starknet.memory programs/fibonacci_starknet.trace
```

We will need the trace and memory files generated.

