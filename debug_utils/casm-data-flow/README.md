# Casm Data Flow

A data flow analyzer for a Cairo VM. Currently, it only supports starknet contracts.

It is useful when trying to debug a gas difference between Cairo Native and Cairo VM.

## Usage

First, we need to compile the contract:

```bash
../../cairo2/bin/starknet-compile -s programs/fibonacci_starknet.cairo > programs/fibonacci_starknet.sierra.json
../../cairo2/bin/starknet-sierra-compile programs/fibonacci_starknet.sierra.json programs/fibonacci_starknet.casm.json
```

Then, we execute the contract. In this example, we are using [our Cairo VM](https://github.com/lambdaclass/cairo-vm).

```bash
cargo run --example run-contract -- programs/fibonacci_starknet.casm.json programs/fibonacci_starknet.memory programs/fibonacci_starknet.trace
```

The example prints the starting and final gas:

```
Starting Gas: 18446744073709551615
Final Gas: 18446744073709544205
```

We will need the trace and memory files generated, to find a path from the starting gas to the final gas.

```bash
cargo run -- --program-path programs/fibonacci_starknet.casm.json --trace-path programs/fibonacci_starknet.trace --memory-path programs/fibonacci_starknet.memory -s 18446744073709551615 -t 18446744073709544205
```

The binary prints the full path from the starting gas to the final gas.

```
  [184] = 18446744073709551615
  [212] = 18446744073709550345 (Δ-1270)
  [222] = 18446744073709549075 (Δ-1270)
  [232] = 18446744073709547805 (Δ-1270)
  [242] = 18446744073709546535 (Δ-1270)
  [252] = 18446744073709545265 (Δ-1270)
  [262] = 18446744073709543995 (Δ-1270)
  [272] = 18446744073709542725 (Δ-1270)
  [282] = 18446744073709541455 (Δ-1270)
  [292] = 18446744073709540185 (Δ-1270)
  [302] = 18446744073709538915 (Δ-1270)
  [312] = 18446744073709537645 (Δ-1270)
  [315] = 18446744073709539715 (Δ2070)
  [321] = 18446744073709544205 (Δ4490)
```
