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

## ⚠️ Disclaimer

🚧 `cairo-native` is still being built therefore API breaking changes might happen often so use it at your own risk. 🚧

For versions under `1.0` `cargo` doesn't comply with [semver](https://semver.org/), so we advise to pin the version the version you use. This can be done by adding `cairo-native = "0.1.0"` to your Cargo.toml

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
1. `box_forward_snapshot`
1. `branch_align`
1. `bytes31_const`
1. `bytes31_to_felt252`
1. `bytes31_try_from_felt252`
1. `call_contract_syscall` (StarkNet)
1. `class_hash_to_felt252` (StarkNet)
1. `class_hash_try_from_felt252` (StarkNet)
1. `const_as_box`
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
1. `enum_from_bounded_int`
1. `enum_init`
1. `enum_match`
1. `enum_snapshot_match`
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
1. `get_available_gas`
1. `get_block_hash_syscall` (StarkNet)
1. `get_builtin_costs` (5)
1. `get_execution_info_syscall` (StarkNet)
1. `hades_permutation`
1. `i128_diff`
1. `i16_diff`
1. `i32_diff`
1. `i64_diff`
1. `i8_diff`
1. `into_box` (2)
1. `jump`
1. `keccak_syscall` (StarkNet)
1. `library_call_syscall` (StarkNet)
1. `match_nullable`
1. `null`
1. `nullable_forward_snapshot`
1. `nullable_from_box`
1. `pedersen`
1. `print`
1. `rename`
1. `replace_class_syscall` (StarkNet)
1. `revoke_ap_tracking`
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
1. `send_message_to_l1_syscall` (StarkNet)
1. `snapshot_take` (1)
1. `span_from_tuple`
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
1. `struct_snapshot_deconstruct`
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
1. coupon
</details>

<details>
<summary>Not yet implemented libfuncs (testing category only, click to open)</summary>
Testing libfuncs:

1. `pop_log` (StarkNet, testing)
1. `redeposit_gas`
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
</details>

Footnotes on the libfuncs list:

1. It is implemented but we're not handling potential issues like lifetimes yet.
1. It is implemented but we're still debating whether it should be a Rust-like `Box<T>` or if it's fine treating it like another variable.
1. It is implemented but side-effects are not yet handled (ex. array cloning/dropping).
1. Not supported by the Cairo to Sierra compiler.
1. Implemented with a dummy. It doesn't do anything yet.

## Getting Started

### Dependencies

- Linux or macOS (aarch64 included) only for now
- LLVM 18 with MLIR: On debian you can use [apt.llvm.org](https://apt.llvm.org/), on macOS you can use brew
- Rust 1.78.0 or later, since we make use of the u128 [abi change](https://blog.rust-lang.org/2024/03/30/i128-layout-update.html).
- Git

### Setup

> This step applies to all operating systems.

Run the following make target to install the dependencies (**both Linux and macOS**):

```bash
make deps
```

#### Linux

Since Linux distributions change widely, you need to install LLVM 18 via your package manager, compile it or check if the current release has a Linux binary.

If you are on Debian/Ubuntu, check out the repository https://apt.llvm.org/
Then you can install with:

```bash
sudo apt-get install llvm-18 llvm-18-dev llvm-18-runtime clang-18 clang-tools-18 lld-18 libpolly-18-dev libmlir-18-dev mlir-18-tools
```

If you decide to build from source, here are some indications:

<details><summary>Install LLVM from source instructions</summary>

```bash
# Go to https://github.com/llvm/llvm-project/releases
# Download the latest LLVM 18 release:
# The blob to download is called llvm-project-18.x.x.src.tar.xz

# For example
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.4/llvm-project-18.1.4.src.tar.xz
tar xf llvm-project-18.1.4.src.tar.xz

cd llvm-project-18.1.4.src.tar
mkdir build
cd build

# The following cmake command configures the build to be installed to /opt/llvm-18
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra;lld;polly" \
   -DLLVM_BUILD_EXAMPLES=OFF \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_INSTALL_PREFIX=/opt/llvm-18 \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_PARALLEL_LINK_JOBS=4 \
   -DLLVM_ENABLE_BINDINGS=OFF \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_ENABLE_ASSERTIONS=OFF

ninja install
```

</details>

Setup a environment variable called `MLIR_SYS_180_PREFIX`, `LLVM_SYS_180_PREFIX` and `TABLEGEN_180_PREFIX` pointing to the llvm directory:

```bash
# For Debian/Ubuntu using the repository, the path will be /usr/lib/llvm-18
export MLIR_SYS_180_PREFIX=/usr/lib/llvm-18
export LLVM_SYS_180_PREFIX=/usr/lib/llvm-18
export TABLEGEN_180_PREFIX=/usr/lib/llvm-18
```

Run the deps target to install the other dependencies such as the cairo compiler (for tests, benchmarks).

```bash
make deps
```

#### MacOS

The makefile `deps` target (which you should have ran before) installs LLVM 18 with brew for you, afterwards you need to execute the `env-macos.sh` script to setup the
needed environment variables.

```bash
source env-macos.sh
```

### Make commands:

Running `make` by itself will list available targets.

- Install the necessary dependencies (on Linux, you need to get LLVM 18 manually):

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

- Compile the runtime library used for ahead of time compilation:

```bash
make runtime
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

Example code to run a Starknet contract:

```rust
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
$MLIR_SYS_180_PREFIX=/path/to/llvm18  # Required for non-standard LLVM install locations.
$LLVM_SYS_180_PREFIX=/path/to/llvm18  # Required for non-standard LLVM install locations.
$TABLEGEN_180_PREFIX=/path/to/llvm18  # Required for non-standard LLVM install locations.
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
"$MLIR_SYS_180_PREFIX/bin/mlir-opt" \
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
"$MLIR_SYS_180_PREFIX"/bin/mlir-translate --mlir-to-llvmir program-llvm.mlir -o program.ll

# compile natively
"$MLIR_SYS_180_PREFIX"/bin/clang program.ll -Wno-override-module \
    -L "$MLIR_SYS_180_PREFIX"/lib -L"./target/release/" \
    -lsierra2mlir_utils -lmlir_c_runner_utils \
    -Wl,-rpath "$MLIR_SYS_180_PREFIX"/lib \
    -Wl,-rpath ./target/release/ \
    -o program

./program
```

# cairo-native-test cli tool

This tool mimics the `cairo-test` [tool](https://github.com/starkware-libs/cairo/tree/main/crates/cairo-lang-test-runner) and is identical to it, the only feature it doesn't have is the profiler.

You can download it on our [releases](https://github.com/lambdaclass/cairo_native/releases) page.

```bash
$ cairo-native-test --help
Compiles a Cairo project and runs all the functions marked as `#[test]`.
Exits with 1 if the compilation or run fails, otherwise 0.

Usage: cairo-native-test [OPTIONS] <PATH>

Arguments:
  <PATH>  The Cairo project path to compile and run its tests

Options:
  -s, --single-file            Whether path is a single file
      --allow-warnings         Allows the compilation to succeed with warnings
  -f, --filter <FILTER>        The filter for the tests, running only tests containing the filter string [default: ]
      --include-ignored        Should we run ignored tests as well
      --ignored                Should we run only the ignored tests
      --starknet               Should we add the starknet plugin to run the tests
      --run-mode <RUN_MODE>    Run with JIT or AOT (compiled) [default: jit] [possible values: aot, jit]
  -O, --opt-level <OPT_LEVEL>  Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3 [default: 0]
  -h, --help                   Print help
  -V, --version                Print version
```

For single files, you can use the `-s, --single-file` option.

For a project, it needs to have a `cairo_project.toml` specifying the crate_roots. You can find an
example under the `cairo-tests/` folder, which is a cairo project that works with this tool.

```
cairo-native-test -s myfile.cairo

cairo-native-test ./cairo-tests/
```

This will run all the tests (functions marked with the `#[test]` attribute).

# scarb-native-test cli tool

This tool mimics the `scarb test` [command](https://github.com/software-mansion/scarb/tree/main/extensions/scarb-cairo-test).
You can download it on our [releases](https://github.com/lambdaclass/cairo_native/releases) page.

```bash
$ scarb-native-test --help
Compiles all packages from a Scarb project matching `packages_filter` and
runs all functions marked with `#[test]`. Exits with 1 if the compilation
or run fails, otherwise 0.

Usage: scarb-native-test [OPTIONS]

Options:
  -p, --package <SPEC>         Packages to run this command on, can be a concrete package name (`foobar`) or a prefix glob (`foo*`) [env: SCARB_PACKAGES_FILTER=] [default: *]
  -w, --workspace              Run for all packages in the workspace
  -f, --filter <FILTER>        Run only tests whose name contain FILTER [default: ]
      --include-ignored        Run ignored and not ignored tests
      --ignored                Run only ignored tests
      --run-mode <RUN_MODE>    Run with JIT or AOT (compiled) [default: jit] [possible values: aot, jit]
  -O, --opt-level <OPT_LEVEL>  Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3 [default: 0]
  -h, --help                   Print help
  -V, --version                Print version
```

## Debugging Tips

### Useful environment variables

These 2 env vars will dump the generated MLIR code from any compilation on the current working directory as:

- `dump.mlir`: The MLIR code after passes without locations.
- `dump-debug.mlir`: The MLIR code after passes with locations.
- `dump-prepass.mlir`: The MLIR code before without locations.
- `dump-prepass-debug.mlir`: The MLIR code before passes with locations.

Do note that the MLIR with locations is in pretty form and thus not suitable to pass to `mlir-opt`.

```bash
export NATIVE_DEBUG_DUMP_PREPASS=1
export NATIVE_DEBUG_DUMP=1
```

Enable logging to see the compilation process:

```bash
export RUST_LOG="cairo_native=trace"
```

Other tips:

- Try to find the minimal program to reproduce an issue, the more isolated the easier to test.
- Use the `debug_utils` print utilities, more info [here](https://lambdaclass.github.io/cairo_native/cairo_native/metadata/debug_utils/struct.DebugUtils.html):

```rust
#[cfg(feature = "with-debug-utils")]
{
    metadata.get_mut::<DebugUtils>()
        .unwrap()
        .print_pointer(context, helper, entry, ptr, location)?;
}
```
