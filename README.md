<div align="center">

### ⚡ Cairo Native ⚡

A compiler to convert Cairo's intermediate representation "Sierra" code <br>
to machine code via MLIR and LLVM.

[Report Bug](https://github.com/lambdaclass/cairo_native/issues/new) · [Request Feature](https://github.com/lambdaclass/cairo_native/issues/new)

[![Telegram Chat][tg-badge]][tg-url]
[![rust](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml/badge.svg)](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/lambdaclass/cairo_native)](https://codecov.io/gh/lambdaclass/cairo_native)
[![license](https://img.shields.io/github/license/lambdaclass/cairo_native)](/LICENSE)
[![pr-welcome]](#-contributing)

[tg-badge]: https://img.shields.io/endpoint?url=https%3A%2F%2Ftg.sumanjay.workers.dev%2FLambdaStarkNet%2F&logo=telegram&label=chat&color=neon
[tg-url]: https://t.me/LambdaStarkNet
[pr-welcome]: https://img.shields.io/static/v1?color=orange&label=PRs&style=flat&message=welcome

</div>

To get started on how to setup and run cairo-native check the [getting started](#getting-started) section.

To read more in-depth documentation, visit [this page](https://lambdaclass.notion.site/Documentation-cd2b99eb14344a40837b1740238a918c).

- [Implemented Library Functions](#implemented-library-functions)
- [Getting started](#getting-started)
- [Command Line Interface](#command-line-interface)
- [Benchmarking](#benchmarking)
- [API usage example](#api--usage-example)
- [From MLIR to native binary](#from-mlir-to-native-binary)

## Implemented Library Functions

Cairo Native works by leveraging the intermediate representation of Cairo called Sierra.
Sierra uses a list of builtin functions that implement the language functionality, those are called library functions, short: **libfuncs**.
Basically every statement in a sierra program is a call to a **libfunc**, thus they are the core of Cairo Native progress towards feature parity.

This is a list of the current progress implementing each **libfunc**.

<details>
<summary>Implemented libfuncs (click to open)</summary>

1. `alloc_local`
1. `array_append`
1. `array_get`
1. `array_len`
1. `array_new`
1. `array_pop_front_consume`
1. `array_pop_front`
1. `array_slice`
1. `array_snapshot_pop_back`
1. `array_snapshot_pop_front`
1. `bitwise`
1. `bool_and_impl`
1. `bool_not_impl`
1. `bool_or_impl`
1. `bool_to_felt252`
1. `bool_xor_impl`
1. `branch_align`
1. `call_contract_syscall` (StarkNet)
1. `class_hash_try_from_felt252` (StarkNet)
1. `contract_address_const` (StarkNet)
1. `contract_address_to_felt252` (StarkNet)
1. `contract_address_try_from_felt252` (StarkNet)
1. `deploy_syscall` (StarkNet)
1. `disable_ap_tracking`
1. `downcast`
1. `drop` (3)
1. `dup` (3)
1. `ec_neg`
1. `ec_point_from_x_nz`
1. `ec_point_is_zero`
1. `ec_point_try_new_nz`
1. `ec_point_unwrap`
1. `ec_point_zero`
1. `ec_state_add_mul`
1. `ec_state_add`
1. `ec_state_init`
1. `ec_state_try_finalize_nz`
1. `emit_event_syscall` (StarkNet)
1. `enable_ap_tracking`
1. `enum_init`
1. `enum_match`
1. `felt252_add_const` (4)
1. `felt252_add`
1. `felt252_const`
1. `felt252_dict_entry_finalize`
1. `felt252_dict_entry_get`
1. `felt252_dict_new`
1. `felt252_dict_squash`
1. `felt252_div_const` (4)
1. `felt252_div` (4)
1. `felt252_is_zero`
1. `felt252_mul_const` (4)
1. `felt252_mul`
1. `felt252_sub_const` (4)
1. `felt252_sub`
1. `finalize_locals`
1. `function_call`
1. `get_block_hash_syscall` (StarkNet)
1. `get_builtin_costs` (5)
1. `get_execution_info_syscall` (StarkNet)
1. `into_box` (2)
1. `jump`
1. `keccak_syscall` (StarkNet)
1. `library_call_syscall` (StarkNet)
1. `match_nullable`
1. `null`
1. `nullable_from_box`
1. `pedersen`
1. `print`
1. `rename`
1. `replace_class_syscall` (StarkNet)
1. `revoke_ap_tracking` (1)
1. `send_message_to_l1_syscall` (StarkNet)
1. `snapshot_take` (6)
1. `storage_address_from_base_and_offset` (StarkNet)
1. `storage_address_from_base` (StarkNet)
1. `storage_address_to_felt252` (StarkNet)
1. `storage_address_try_from_felt252` (StarkNet)
1. `storage_base_address_const` (StarkNet)
1. `storage_base_address_from_felt252` (StarkNet)
1. `storage_read_syscall` (StarkNet)
1. `storage_write_syscall` (StarkNet)
1. `store_local`
1. `store_temp`
1. `struct_construct`
1. `struct_deconstruct`
1. `u128_byte_reverse`
1. `u128_const`
1. `u128_eq`
1. `u128_guarantee_mul`
1. `u128_is_zero`
1. `u128_mul_guarantee_verify`
1. `u128_overflowing_add`
1. `u128_overflowing_sub`
1. `u128_safe_divmod`
1. `u128_sqrt`
1. `u128_to_felt252`
1. `u128s_from_felt252`
1. `u16_const`
1. `u16_eq`
1. `u16_is_zero`
1. `u16_overflowing_add`
1. `u16_overflowing_sub`
1. `u16_safe_divmod`
1. `u16_sqrt`
1. `u16_to_felt252`
1. `u16_try_from_felt252`
1. `u16_wide_mul`
1. `u256_is_zero`
1. `u256_safe_divmod`
1. `u256_sqrt`
1. `u32_const`
1. `u32_eq`
1. `u32_is_zero`
1. `u32_overflowing_add`
1. `u32_overflowing_sub`
1. `u32_safe_divmod`
1. `u32_sqrt`
1. `u32_to_felt252`
1. `u32_try_from_felt252`
1. `u32_wide_mul`
1. `u64_const`
1. `u64_eq`
1. `u64_is_zero`
1. `u64_overflowing_add`
1. `u64_overflowing_sub`
1. `u64_safe_divmod`
1. `u64_sqrt`
1. `u64_to_felt252`
1. `u64_try_from_felt252`
1. `u64_wide_mul`
1. `u8_const`
1. `u8_eq`
1. `u8_is_zero`
1. `u8_overflowing_add`
1. `u8_overflowing_sub`
1. `u8_safe_divmod`
1. `u8_sqrt`
1. `u8_to_felt252`
1. `u8_try_from_felt252`
1. `u8_wide_mul`
1. `unbox` (2)
1. `unwrap_non_zero`
1. `upcast`
1. `withdraw_gas_all` (5)
1. `withdraw_gas` (5)
</details>

<details>
<summary>Not yet implemented libfuncs (click to open)</summary>

1. `class_hash_const` (StarkNet)
1. `class_hash_to_felt252` (StarkNet)
1. `enum_snapshot_match`
1. `get_available_gas`
1. `pop_log` (StarkNet, testing)
1. `poseidon`
1. `redeposit_gas`
1. `secp256k1_add_syscall` (StarkNet)
1. `secp256k1_get_point_from_x_syscall` (StarkNet)
1. `secp256k1_get_xy_syscall` (StarkNet)
1. `secp256k1_mul_syscall` (StarkNet)
1. `secp256k1_new_syscall` (StarkNet)
1. `secp256r1_add_syscall` (StarkNet)
1. `secp256r1_get_point_from_x_syscall` (StarkNet)
1. `secp256r1_get_xy_syscall` (StarkNet)
1. `secp256r1_mul_syscall` (StarkNet)
1. `secp256r1_new_syscall` (StarkNet)
1. `set_account_contract_address` (StarkNet, testing)
1. `set_block_number` (StarkNet, testing)
1. `set_block_timestamp` (StarkNet, testing)
1. `set_caller_address` (StarkNet, testing)
1. `set_chain_id` (StarkNet, testing)
1. `set_contract_address` (StarkNet, testing)
1. `set_max_fee` (StarkNet, testing)
1. `set_nonce` (StarkNet, testing)
1. `set_sequencer_address` (StarkNet, testing)
1. `set_signature` (StarkNet, testing)
1. `set_transaction_hash` (StarkNet, testing)
1. `set_version` (StarkNet, testing)
1. `struct_snapshot_deconstruct`
1. `u512_safe_divmod_by_u256`
1. `i128_diff`
1. `i16_diff`
1. `i32_diff`
1. `i64_diff`
1. `i8_diff`

</details>

Footnotes on the libfuncs list:

1. It is implemented but we're not sure if it has some stuff we don't know of.
2. It is implemented but we're still debating whether it should be a Rust-like `Box<T>` or if it's fine treating it like another variable.
3. It is implemented but side-effects are not yet handled (ex. array cloning/dropping).
4. Not supported by the Cairo to Sierra compiler.
5. Implemented with a dummy. It doesn't do anything yet.
6. It is implemented but we're not handling potential issues like lifetimes yet.

## Getting Started

### Dependencies

- Linux or macOS (aarch64 included) only for now
- LLVM 17 with MLIR: On debian you can use [apt.llvm.org](https://apt.llvm.org/), on macOS you can use brew
- Nightly Rust
- Git

### Setup

> This step applies to all operating systems.

Run the following make target to install the dependencies (**both Linux and macOS**):

```bash
make deps
```

#### Linux

Since Linux distributions change widely, you need to install LLVM 17 via your package manager, compile it or check if the current release has a Linux binary.

If you are on Debian/Ubuntu, check out the repository https://apt.llvm.org/
Then you can install with:

```bash
sudo apt-get install llvm-17 llvm-17-dev llvm-17-runtime clang-17 clang-tools-17 lld-17 libpolly-17-dev libmlir-17-dev mlir-17-tools
```

If you decide to build from source, here are some indications:

<details><summary>Install LLVM from source instructions</summary>

```bash
# Go to https://github.com/llvm/llvm-project/releases
# Download the latest LLVM 17 release:
# The blob to download is called llvm-project-17.x.x.src.tar.xz

# For example
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.3/llvm-project-17.0.3.src.tar.xz
tar xf llvm-project-17.0.3.src.tar.xz

cd llvm-project-17.0.3.src.tar
mkdir build
cd build

# The following cmake command configures the build to be installed to /opt/llvm-17
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra;lld;polly" \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_INSTALL_PREFIX=/opt/llvm-17 \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_PARALLEL_LINK_JOBS=4 \
   -DLLVM_ENABLE_BINDINGS=OFF \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_ENABLE_ASSERTIONS=OFF

ninja install
```

</details>

Setup a environment variable called `MLIR_SYS_170_PREFIX` and `TABLEGEN_170_PREFIX` pointing to the llvm directory:

```bash
# For Debian/Ubuntu using the repository, the path will be /usr/lib/llvm-17
export MLIR_SYS_170_PREFIX=/usr/lib/llvm-17
export TABLEGEN_170_PREFIX=/usr/lib/llvm-17
```

Run the deps target to install the other dependencies such as the cairo compiler (for tests, benchmarks).
```bash
make deps
```

#### MacOS

The makefile `deps` target (which you should have ran before) installs LLVM 17 with brew for you, afterwards you need to execute the `env-macos.sh` script to setup the
needed environment variables.

```bash
source env-macos.sh
```

### Make commands:

Running `make` by itself will list available targets.

- Install the necessary dependencies (on Linux, you need to get LLVM 17 manually):

```bash
make deps
```

- Build a release version:

```bash
make build
```

Or with your native CPU Architecture for even more performance (usually):
```bash
make build-native
```

- Install the `cairo-native-dump` and `cairo-native-run` commands:

```bash
make install
```

- Build a optimized development version:

```bash
make build-dev
```

- View and open the docs:

```bash
make doc-open
```

- Run the tests:

```bash
make test
```

- Generate coverage:

```bash
make coverage
```

- Run clippy and format checks:

```bash
make check
```

## Command Line Interface

`cairo-native-dump`:
```
Usage: cairo-native-dump [OPTIONS] <INPUT>

Arguments:
  <INPUT>

Options:
  -o, --output <OUTPUT>  [default: -]
  -h, --help             Print help
```

`cairo-native-run`:

This tool allows to run programs using the JIT engine, like the `cairo-run` tool, the parameters can only be felt values.

`echo '1' | cairo-native-run 'program.cairo' 'program::program::main' --inputs - --outputs -`

```
Usage: cairo-native-run [OPTIONS] <INPUT> <ENTRY_POINT>

Arguments:
  <INPUT>
  <ENTRY_POINT>

Options:
  -i, --inputs <INPUTS>
  -o, --outputs <OUTPUTS>
  -p, --print-outputs
  -h, --help               Print help
```

# API usage example

This is a usage example using the API for an easy Cairo program that requires the least setup to get running. It allows you to compile and execute a program using the JIT.

Example code to run a program:

```rust
use cairo_felt::Felt252;
use cairo_native::context::NativeContext;
use cairo_native::executor::NativeJitEngine;
use cairo_native::values::JITValue;
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
    let params = &[JITValue::Felt252(Felt252::from_bytes_be(b"user"))];

    // Find the entry point id by its name.
    let entry_point = "hello::hello::greet";
    let entry_point_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);

    // Instantiate the executor.
    let native_executor = NativeJitEngine::new(native_program);

    // Execute the program.
    let result = native_executor
        .execute(entry_point_id, params, None)
        .unwrap();

    println!("Cairo program was compiled and executed successfully.");
    println!("{:?}", result);
}
```

Example code to run a Starknet contract:

```rust
use cairo_felt::Felt252;
use cairo_lang_compiler::CompilerConfig;
use cairo_lang_starknet::contract_class::compile_path;
use cairo_native::context::NativeContext;
use cairo_native::executor::NativeJitEngine;
use cairo_native::utils::find_entry_point_by_idx;
use cairo_native::values::JITValue;
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

    let native_executor = NativeJitEngine::new(native_program);

    let result = native_executor
        .execute_contract(
            fn_id,
            // The calldata
            &[JITValue::Felt252(Felt252::new(1))],
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
    ) -> SyscallResult<cairo_felt::Felt252> {
        println!("Called `get_block_hash({block_number})` from MLIR.");
        Ok(Felt252::from_bytes_be(b"get_block_hash ok"))
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
        class_hash: cairo_felt::Felt252,
        contract_address_salt: cairo_felt::Felt252,
        calldata: &[cairo_felt::Felt252],
        deploy_from_zero: bool,
        _gas: &mut u128,
    ) -> SyscallResult<(cairo_felt::Felt252, Vec<cairo_felt::Felt252>)> {
        println!("Called `deploy({class_hash}, {contract_address_salt}, {calldata:?}, {deploy_from_zero})` from MLIR.");
        Ok((
            class_hash + contract_address_salt,
            calldata.iter().map(|x| x + &Felt252::new(1)).collect(),
        ))
    }

    fn replace_class(
        &mut self,
        class_hash: cairo_felt::Felt252,
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `replace_class({class_hash})` from MLIR.");
        Ok(())
    }

    fn library_call(
        &mut self,
        class_hash: cairo_felt::Felt252,
        function_selector: cairo_felt::Felt252,
        calldata: &[cairo_felt::Felt252],
        _gas: &mut u128,
    ) -> SyscallResult<Vec<cairo_felt::Felt252>> {
        println!(
            "Called `library_call({class_hash}, {function_selector}, {calldata:?})` from MLIR."
        );
        Ok(calldata.iter().map(|x| x * &Felt252::new(3)).collect())
    }

    fn call_contract(
        &mut self,
        address: cairo_felt::Felt252,
        entry_point_selector: cairo_felt::Felt252,
        calldata: &[cairo_felt::Felt252],
        _gas: &mut u128,
    ) -> SyscallResult<Vec<cairo_felt::Felt252>> {
        println!(
            "Called `call_contract({address}, {entry_point_selector}, {calldata:?})` from MLIR."
        );
        Ok(calldata.iter().map(|x| x * &Felt252::new(3)).collect())
    }

    fn storage_read(
        &mut self,
        address_domain: u32,
        address: cairo_felt::Felt252,
        _gas: &mut u128,
    ) -> SyscallResult<cairo_felt::Felt252> {
        println!("Called `storage_read({address_domain}, {address})` from MLIR.");
        Ok(address * &Felt252::new(3))
    }

    fn storage_write(
        &mut self,
        address_domain: u32,
        address: cairo_felt::Felt252,
        value: cairo_felt::Felt252,
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `storage_write({address_domain}, {address}, {value})` from MLIR.");
        Ok(())
    }

    fn emit_event(
        &mut self,
        keys: &[cairo_felt::Felt252],
        data: &[cairo_felt::Felt252],
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `emit_event({keys:?}, {data:?})` from MLIR.");
        Ok(())
    }

    fn send_message_to_l1(
        &mut self,
        to_address: cairo_felt::Felt252,
        payload: &[cairo_felt::Felt252],
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
        Ok(U256(Felt252::from(1234567890).to_le_bytes()))
    }

    /*
    ... more code here, check out the full example in examples/starknet.rsd
    */
}

```

For more examples, check out the `examples/` directory.

## Benchmarking

### Requirements

- [hyperfine](https://github.com/sharkdp/hyperfine): `cargo install hyperfine`
- [cairo >=1.0](https://github.com/starkware-libs/cairo)
- Cairo Corelibs
- LLVM 16 with MLIR

You need to setup some environment variables:
```bash
$MLIR_SYS_170_PREFIX=/path/to/llvm17  # Required for non-standard LLVM install locations.
$TABLEGEN_170_PREFIX=/path/to/llvm17  # Required for non-standard LLVM install locations.
```

```bash
make bench
```

The `bench` target will run the `./scripts/bench-hyperfine.sh` script.
This script runs hyperfine commands to compare the execution time of programs in the `./programs/benches/` folder.
Each program is compiled and executed via the execution engine with the `cairo-native-run` command and via the cairo-vm with the `cairo-run` command provided by the `cairo` codebase.
The `cairo-run` command should be available in the `$PATH` and ideally compiled with `cargo build --release`.
If you want the benchmarks to run using a specific build, or the `cairo-run` commands conflicts with something (e.g. the cairo-svg package binaries in macos) then the command to run `cairo-run` with a full path can be specified with the `$CAIRO_RUN` environment variable.

## From MLIR to native binary
```bash
# to mlir with llvm dialect
sierra2mlir program.sierra -o program.mlir

# translate all dialects to the llvm dialect
"$MLIR_SYS_170_PREFIX/bin/mlir-opt" \
        --canonicalize \
        --convert-scf-to-cf \
        --canonicalize \
        --cse \
        --expand-strided-metadata \
        --finalize-memref-to-llvm \
        --convert-func-to-llvm \
        --convert-index-to-llvm \
        --reconcile-unrealized-casts \
        "program.mlir" \
        -o "program-llvm.mlir"

# translate mlir to llvm-ir
"$MLIR_SYS_170_PREFIX"/bin/mlir-translate --mlir-to-llvmir program-llvm.mlir -o program.ll

# compile natively
"$MLIR_SYS_170_PREFIX"/bin/clang program.ll -Wno-override-module \
    -L "$MLIR_SYS_170_PREFIX"/lib -L"./target/release/" \
    -lsierra2mlir_utils -lmlir_c_runner_utils \
    -Wl,-rpath "$MLIR_SYS_170_PREFIX"/lib \
    -Wl,-rpath ./target/release/ \
    -o program

./program
```
