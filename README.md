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
- [Getting started](getting-started)
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
- LLVM 16+ with MLIR: On debian you can use [apt.llvm.org](https://apt.llvm.org/), on macOS you can use brew
- Nightly Rust
- Git

### Setup

Install LLVM with MLIR. You can use the official packages provided by LLVM.

#### Linux

Setup a environment variable called `MLIR_SYS_160_PREFIX` pointing to the llvm directory:

```bash
export MLIR_SYS_160_PREFIX=/usr/lib/llvm-16
```

#### MacOS

```bash
brew install llvm@16
export MLIR_SYS_160_PREFIX=/opt/homebrew/opt/llvm@16
```

### Make commands:

Running `make` by itself will list available targets.

- Build a release version:

```bash
make build
```

Or with your native CPU Architecture for even more perfomance (usually):
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
```
Usage: cairo-native-run [OPTIONS] <INPUT> <ENTRY_POINT>

Arguments:
  <INPUT>
  <ENTRY_POINT>

Options:
  -i, --inputs <INPUTS>
  -o, --outputs <OUTPUTS>
  -p, --print-outputs
  -g, --available-gas <AVAILABLE_GAS>
  -h, --help                           Print help
```

# API usage example

This is a usage example using the API for an easy Cairo program that requires the least setup to get running. It allows you to compile and execute a program using the JIT. The inputs and outputs in this case are
serialized using `serde_json` (so the format is JSON).

Do note that, unlike `cairo-run`, `cairo-native` needs all inputs to be passed, even the implicit ones such as builtins. Some examples of these are the `GasBuiltin` (that is basically the gas), `RangeCheck`, etc. Most of them, with the exception of `GasBuiltin` can simply be
passed as `null`.

If the wrong inputs are passed, an error reports the needed inputs. You can also check the needed inputs by
compiling the program to sierra and checking the arguments of the entry point you chose, it will look something like:

```
example::example::main@0([0]: Pedersen, [1]: felt252, [2]: felt252) -> (Pedersen, felt252);
```

In this case, we take the pedersen builtin and 2 felts. So we pass the following json:

```json
[null, [1,0,0,0,0,0,0,0],  [2,0,0,0,0,0,0,0]]
```

The first `null` is the pedersen builtin, in cairo-native most builtins (with the exception of `GasBuiltin`) are not used at all, so `null` works.

The two following inputs are felts encoded as a u32 array of length 8 in little endian order.
You can use the functions provided on the `cairo_native::utils` module `felt252_str`, `felt252_bigint` and `felt252_short_str` to easily encode felts to this format.


Example code:

```rust
use cairo_native::context::NativeContext;
use cairo_native::executor::NativeExecutor;
use serde_json::json;
use std::{io::stdout, path::Path};

fn main() {
    // FIXME: Remove when cairo adds an easy to use API for setting the corelibs path.
    std::env::set_var(
        "CARGO_MANIFEST_DIR",
        format!("{}/a", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
    );

    #[cfg(not(feature = "with-runtime"))]
    compile_error!("This example requires the `with-runtime` feature to be active.");

    let program_path = Path::new("programs/examples/hello.cairo");
    // Compile the cairo program to sierra.
    let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);

    // Instantiate a Cairo Native MLIR contex. This data structure is responsible for the
    // MLIR initialization and compilation of sierra programs into a MLIR module.
    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_program = native_context.compile(&sierra_program).unwrap();

    // Get necessary information for the execution of the program from a given entrypoint:
    //   * entrypoint function id
    //   * required initial gas
    let name = cairo_native::utils::felt252_short_str("user");
    let entry_point = "hello::hello::greet";
    let params = json!([name]);
    let returns = &mut serde_json::Serializer::new(stdout());
    let fn_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);
    let required_init_gas = native_program.get_required_init_gas(&fn_id);

    // Instantiate MLIR executor.
    let native_executor = NativeExecutor::new(native_program);

    // Execute the program
    native_executor
        .execute(&fn_id, params, returns, required_init_gas).unwrap();

    println!("Cairo program was compiled and executed succesfully.");
}
```

## Benchmarking

### Requirements

- [hyperfine](https://github.com/sharkdp/hyperfine): `cargo install hyperfine`
- [cairo >=1.0](https://github.com/starkware-libs/cairo)
- Cairo Corelibs
- LLVM 16 with MLIR

You need to setup some environment variables:
```bash
$MLIR_SYS_160_PREFIX=/path/to/llvm16  # Required for non-standard LLVM install locations.
```

```bash
make bench
```

The `bench` target will run the `./scripts/bench-hyperfine.sh` script.
This script runs hyperfine comands to compare the execution time of programs in the `./programs/benches/` folder.
Each program is compiled and executed via the execution engine with the `sierrajit` command and via the cairo-vm with the `cairo-run` command provided by the `cairo` codebase.
The `cairo-run` command should be available in the `$PATH` and ideally compiled with `cargo build --release`.
If you want the benchmarks to run using a specific build, or the `cairo-run` commands conflicts with something (e.g. the cairo-svg package binaries in macos) then the command to run `cairo-run` with a full path can be specified with the `$CAIRO_RUN` environment variable.

## From MLIR to native binary
```bash
# to mlir with llvm dialect
sierra2mlir program.sierra -o program.mlir

# translate mlir to llvm-ir
"$MLIR_SYS_160_PREFIX"/bin/mlir-translate --mlir-to-llvmir program.mlir -o program.ll

# compile natively
"$MLIR_SYS_160_PREFIX"/bin/clang program.ll -Wno-override-module \
    -L "$MLIR_SYS_160_PREFIX"/lib -L"./target/release/" \
    -lsierra2mlir_utils -lmlir_c_runner_utils \
    -Wl,-rpath "$MLIR_SYS_160_PREFIX"/lib \
    -Wl,-rpath ./target/release/ \
    -o program

./program
```
