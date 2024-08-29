# Overview

This crate is a compiler and JIT engine that transforms Sierra (or Cairo) 
sources into MLIR, which can be 
[JIT-executed](https://en.wikipedia.org/wiki/Just-in-time_compilation) or further
compiled into a binary
[ahead of time](https://en.wikipedia.org/wiki/Ahead-of-time_compilation).

## Getting started as a developer
First make sure you have a working environment and are able to compile the 
project without issues. Make sure to follow the [setup](/README.md#setup) guide 
on steps on how to do this.

It is generally recommended to use the `optimized-dev` cargo profile when 
testing or running programs, the make target `make build-dev` will be useful for 
this.

### Other tools
In addition to the tools included in Cairo Native, it is also recommended you 
have `cairo-compile` and `cairo-run` installed to check how the generated sierra 
code looks like, and to compare results manually (when required) which will help 
greatly when implementing functionality into Cairo Native.

You can check the [cairo](https://github.com/starkware-libs/cairo) repository 
for more info on how to get those tools.

## Basic Workflow
After having implemented your desired feature or bug fix, you should check it 
passes all tests and lints, also make sure to add any needed test cases for the 
added code.

```bash
# Check it passes all lints
make check

# Check it passes all tests
make test
```

Then you are free to go and make a PR!

## High level project overview
This will explain how the project is structured, without going into much details 
yet:

### Project dependencies
The major dependencies of the project are the following:
- Melior: This is the crate that abstracts away most of the interfacing with
  MLIR, our compilation target, it uses mlir-sys and tries to safely
  abstract MLIR in Rust.
- Cairo: We use the cairo crates to keep a close tie to the API contracts
  of the language, they provide a really nice way to know what features the
  language has and aids with codegen. For example, most library functions
  are under enumerations, and thanks to Rust exhaustive pattern matching we
  can't miss any.
- Runtime: The JIT runner and compiler depend on a "runtime" that lives on
  this repository too, it aids with more complex stuff like `pedersen`,
  `keccak` and dictionaries that would be quite complex to implement from
  the ground up in MLIR (Basically would be like coding a complex hash
  function in pseudo assembly).

### Common definitions
Within this project there are lots of functions with the same signature.
As their arguments have all the same meaning, they are documented here:

- `context: NativeContext`: The MLIR context.
- `module: &NativeModule`: The compiled MLIR program, with other relevant
  information such as program registry and metadata.
- `program: &Program`: The Sierra input program.
- `registry: &ProgramRegistry<TType, TLibfunc>`: The registry extracted
  from the program.
- `metadata: &mut MetadataStorage`: Current compiler metadata.

## Project layout
The code is laid out in the following sections:

```txt
 src
 ├─ arch.rs             Trampoline assembly for calling functions with dynamic signatures.
 ├─ arch/               Architecture-specific code for the trampoline.
 ├─ bin/                Binary programs
 ├─ block_ext.rs        A melior (MLIR) block trait extension to write less code.
 ├─ cache.rs            Types and implementations of compiled program caches. 
 ├─ compiler.rs         The glue code of the compiler, has the codegen for
                        the function signatures and calls the libfunc
                        codegen implementations.
 ├─ context.rs          The MLIR context wrapper, provides the compile method.
 ├─ debug.rs            
 ├─ docs.rs             Documentation modules.
 ├─ error.rs            Error handling,
 ├─ execution_result.rs Program result parsing.
 ├─ executor.rs         The executor & related code,
 ├─ ffi.cpp             Missing FFI C wrappers,
 ├─ ffi.rs              Missing FFI C wrappers, rust side.
 ├─ lib.rs              The main lib file.
 ├─ libfuncs.rs         Cairo Sierra libfunc glue code & implementations,
 ├─ metadata.rs         Metadata injector to use within the compilation process.
 ├─ module.rs           The MLIR module wrapper.
 ├─ starknet.rs         Starknet syscall handler glue code.
 ├─ starknet_stub.rs    
 ├─ types.rs            Cairo to MLIR type information,
 ├─ utils.rs            Internal utilities.
 └─ values.rs           JIT serialization.
```

### Library functions
Path: `src/libfuncs`

Here are stored all the library function implementations in MLIR, this
contains the majority of the code.

To store information about the different types of library functions sierra
has, we divide them into the following using the enum `SierraLibFunc`:
- **Branching**: These functions are implemented inline, adding blocks and
  jumping as necessary based on given conditions.
- **Constant**: A constant value, this isn't represented as a function and
  is inserted inline.
- **Function**: Any other function.
- **InlineDataFlow**: Functions that can be implemented inline without much
  problem. For example: `dup`, `store_temp`

### Statements
Path: `src/statements`

Here is the code that processes the statements of non-library functions.
It handles dataflow, branching, function calls, variable storage and also
has implementations for the inline library functions.

### User functions
These are extra utility functions unrelated to sierra that aid in the
development, such as wrapping return values and printing them.

## Basic API usage example
The API contains two structs, `NativeContext` and `NativeExecutor`.
The main purpose of `NativeContext` is MLIR initialization, compilation and
lowering to LLVM.
`NativeExecutor` in the other hand is responsible of executing MLIR
compiled sierra programs from an entrypoint. Programs and JIT states can be
cached in contexts where their execution will be done multiple times.

```rust,ignore
use starknet_types_core::felt::Felt;
use cairo_native::context::NativeContext;
use cairo_native::executor::JitNativeExecutor;
use cairo_native::values::JitValue;
use std::path::Path;

let program_path = Path::new("programs/examples/hello.cairo");
// Compile the cairo program to sierra.
let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);

// Instantiate a Cairo Native MLIR context. This data structure is responsible for the MLIR
// initialization and compilation of sierra programs into a MLIR module.
let native_context = NativeContext::new();

// Compile the sierra program into a MLIR module.
let native_program = native_context.compile(&sierra_program, None).unwrap();

// The parameters of the entry point.
let params = &[JitValue::Felt252(Felt::from_bytes_be_slice(b"user"))];

// Find the entry point id by its name.
let entry_point = "hello::hello::greet";
let entry_point_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);

// Instantiate the executor.
let native_executor = JitNativeExecutor::from_native_module(native_program, Default::default());

// Execute the program.
let result = native_executor
    .invoke_dynamic(entry_point_id, params, None)
    .unwrap();

println!("Cairo program was compiled and executed successfully.");
println!("{:?}", result);
```

## Running a Cairo program
This is a usage example using the API for an easy Cairo program that
requires the least setup to get running. It allows you to compile and
execute a program using the JIT.

Example code to run a program:

```rust,ignore
use starknet_types_core::felt::Felt;
use cairo_native::context::NativeContext;
use cairo_native::executor::NativeExecutor;
use cairo_native::values::JitValue;
use std::path::Path;

fn main() {
    let program_path = Path::new("programs/examples/hello.cairo");
    // Compile the cairo program to sierra.
    let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);

    // Instantiate a Cairo Native MLIR context. This data structure is responsible for the MLIR
    // initialization and compilation of sierra programs into a MLIR module.
    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_program = native_context.compile(&sierra_program).unwrap();

    // The parameters of the entry point.
    let params = &[JitValue::Felt252(Felt::from_bytes_be_slice(b"user"))];

    // Find the entry point id by its name.
    let entry_point = "hello::hello::greet";
    let entry_point_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);

    // Instantiate the executor.
    let native_executor = NativeExecutor::new(native_program);

    // Execute the program.
    let result = native_executor
        .execute(entry_point_id, params, None)
        .unwrap();

    println!("Cairo program was compiled and executed successfully.");
    println!("{:?}", result);
}
```

## Running a Starknet contract

Example code to run a Starknet contract:

```rust,ignore
use starknet_types_core::felt::Felt;
use cairo_lang_compiler::CompilerConfig;
use cairo_lang_starknet::contract_class::compile_path;
use cairo_native::context::NativeContext;
use cairo_native::executor::NativeExecutor;
use cairo_native::utils::find_entry_point_by_idx;
use cairo_native::values::JitValue;
use cairo_native::{
    metadata::syscall_handler::SyscallHandlerMeta,
    starknet::{BlockInfo, ExecutionInfo, StarkNetSyscallHandler, SyscallResult, TxInfo, U256},
};
use std::path::Path;

/// To run a starknet contract, we need to use a syscall handler, here we show how to implement one (at the end).
#[derive(Debug)]
struct SyscallHandler;

fn main() {
    let path = Path::new("programs/examples/hello_starknet.cairo");

    let contract = compile_path(
        path,
        None,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let entry_point = contract.entry_points_by_type.constructor.get(0).unwrap();
    let sierra_program = contract.extract_sierra_program().unwrap();

    let native_context = NativeContext::new();

    let mut native_program = native_context.compile(&sierra_program).unwrap();
    native_program
        .insert_metadata(SyscallHandlerMeta::new(&mut SyscallHandler))
        .unwrap();

    // Call the echo function from the contract using the generated wrapper.
    let entry_point_fn =
        find_entry_point_by_idx(&sierra_program, entry_point.function_idx).unwrap();

    let fn_id = &entry_point_fn.id;

    let native_executor = NativeExecutor::new(native_program);

    let result = native_executor
        .execute_contract(
            fn_id,
            // The calldata
            &[JitValue::Felt252(Felt::ONE)],
            u64::MAX.into(),
        )
        .expect("failed to execute the given contract");

    println!();
    println!("Cairo program was compiled and executed successfully.");
    println!("{result:#?}");
}

// Implement an example syscall handler.
impl StarkNetSyscallHandler for SyscallHandler {
    fn get_block_hash(
        &mut self,
        block_number: u64,
        _gas: &mut u128,
    ) -> SyscallResult<Felt> {
        println!("Called `get_block_hash({block_number})` from MLIR.");
        Ok(Felt::from_bytes_be_slice(b"get_block_hash ok"))
    }

    fn get_execution_info(
        &mut self,
        _gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
        println!("Called `get_execution_info()` from MLIR.");
        Ok(ExecutionInfo {
            block_info: BlockInfo {
                block_number: 1234,
                block_timestamp: 2345,
                sequencer_address: 3456.into(),
            },
            tx_info: TxInfo {
                version: 4567.into(),
                account_contract_address: 5678.into(),
                max_fee: 6789,
                signature: vec![1248.into(), 2486.into()],
                transaction_hash: 9876.into(),
                chain_id: 8765.into(),
                nonce: 7654.into(),
            },
            caller_address: 6543.into(),
            contract_address: 5432.into(),
            entry_point_selector: 4321.into(),
        })
    }

    fn deploy(
        &mut self,
        class_hash: Felt,
        contract_address_salt: Felt,
        calldata: &[Felt],
        deploy_from_zero: bool,
        _gas: &mut u128,
    ) -> SyscallResult<(Felt, Vec<Felt>)> {
        println!("Called `deploy({class_hash}, {contract_address_salt}, {calldata:?}, {deploy_from_zero})` from MLIR.");
        Ok((
            class_hash + contract_address_salt,
            calldata.iter().map(|x| x + &Felt::ONE).collect(),
        ))
    }

    fn replace_class(
        &mut self,
        class_hash: Felt,
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `replace_class({class_hash})` from MLIR.");
        Ok(())
    }

    fn library_call(
        &mut self,
        class_hash: Felt,
        function_selector: Felt,
        calldata: &[Felt],
        _gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        println!(
            "Called `library_call({class_hash}, {function_selector}, {calldata:?})` from MLIR."
        );
        Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
    }

    fn call_contract(
        &mut self,
        address: Felt,
        entry_point_selector: Felt,
        calldata: &[Felt],
        _gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        println!(
            "Called `call_contract({address}, {entry_point_selector}, {calldata:?})` from MLIR."
        );
        Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
    }

    fn storage_read(
        &mut self,
        address_domain: u32,
        address: Felt,
        _gas: &mut u128,
    ) -> SyscallResult<Felt> {
        println!("Called `storage_read({address_domain}, {address})` from MLIR.");
        Ok(address * Felt::from(3))
    }

    fn storage_write(
        &mut self,
        address_domain: u32,
        address: Felt,
        value: Felt,
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `storage_write({address_domain}, {address}, {value})` from MLIR.");
        Ok(())
    }

    fn emit_event(
        &mut self,
        keys: &[Felt],
        data: &[Felt],
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `emit_event({keys:?}, {data:?})` from MLIR.");
        Ok(())
    }

    fn send_message_to_l1(
        &mut self,
        to_address: Felt,
        payload: &[Felt],
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `send_message_to_l1({to_address}, {payload:?})` from MLIR.");
        Ok(())
    }

    fn keccak(
        &mut self,
        input: &[u64],
        _gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::U256> {
        println!("Called `keccak({input:?})` from MLIR.");
        Ok(U256(Felt::from(1234567890).to_le_bytes()))
    }

    /*
    ... more code here, check out the full example in examples/starknet.rs
    */
}

```

For more examples, check out the `examples/` directory.
