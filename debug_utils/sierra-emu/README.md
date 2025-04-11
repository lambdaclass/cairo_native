<div align="center">

### ⚡ Cairo Sierra Emulator ⚡

An Cairo emulator directly using the Cairo's intermediate representation "Sierra" instead of CASM.<br>
An useful usecase is to aid in debugging other Cairo related VMs.

[Report Bug](https://github.com/lambdaclass/sierra-emu/issues/new) · [Request Feature](https://github.com/lambdaclass/sierra-emu/issues/new)

[![Telegram Chat][tg-badge]][tg-url]
[![rust](https://github.com/lambdaclass/sierra-emu/actions/workflows/ci.yml/badge.svg)](https://github.com/lambdaclass/sierra-emu/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/lambdaclass/sierra-emu)](https://codecov.io/gh/lambdaclass/sierra-emu)
[![license](https://img.shields.io/github/license/lambdaclass/sierra-emu)](/LICENSE)
[![pr-welcome]](#-contributing)


[tg-badge]: https://img.shields.io/endpoint?url=https%3A%2F%2Ftg.sumanjay.workers.dev%2FLambdaStarkNet%2F&logo=telegram&label=chat&color=neon
[tg-url]: https://t.me/LambdaStarkNet
[pr-welcome]: https://img.shields.io/static/v1?color=orange&label=PRs&style=flat&message=welcome

</div>



## Running the Program
`cargo run <SIERRA PROGRAM> <ENTRYPOINT ID>`

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
