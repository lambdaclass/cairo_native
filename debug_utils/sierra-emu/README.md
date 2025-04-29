# ⚡ Cairo Sierra Emulator ⚡

An Cairo emulator directly using the Cairo's intermediate representation "Sierra" instead of CASM.
An useful usecase is to aid in debugging other Cairo related VMs.

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

With a contract:

```rust
use std::path::Path;

use cairo_lang_compiler::CompilerConfig;
use cairo_lang_starknet::compile::compile_path;
use sierra_emu::{
    starknet::StubSyscallHandler, ContractExecutionResult,
    VirtualMachine,
};
let path = Path::new("programs/hello_starknet.cairo");

let contract = compile_path(
    path,
    None,
    CompilerConfig {
        replace_ids: true,
        ..Default::default()
    },
)
.unwrap();

let sierra_program = contract.extract_sierra_program().unwrap();

let entry_point = contract.entry_points_by_type.external.first().unwrap();

let mut vm = VirtualMachine::new_starknet(
    sierra_program.clone().into(),
    &contract.entry_points_by_type,
);

let calldata = [2.into()];
let initial_gas = 1000000;

// Change the StubSyscallHandler with your own implementation.
let syscall_handler = &mut StubSyscallHandler::default();

vm.call_contract(entry_point.selector.clone().into(), initial_gas, calldata);

let _race = vm.run_with_trace(syscall_handler);
let trace_str = serde_json::to_string_pretty(&trace).unwrap();
std::fs::write("contract_trace.json", trace_str).unwrap();
```

With a program:

```rust
let path = Path::new(path);

let sierra_program = Arc::new(
    compile_cairo_project_at_path(
        path,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap(),
);

let function = find_entry_point_by_name(&sierra_program, func_name).unwrap();

let mut vm = VirtualMachine::new(sierra_program.clone());

let args = args.iter().cloned();
let initial_gas = 1000000;

vm.call_program(function, initial_gas, args);

let syscall_handler = &mut StubSyscallHandler::default();
let trace = vm.run_with_trace(syscall_handler);
```
