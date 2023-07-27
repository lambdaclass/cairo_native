# Cairo Native
[![test](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml/badge.svg)](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml)
[![mdbook](https://img.shields.io/badge/mdbook-link-blue)](https://lambdaclass.github.io/cairo_native/)

A compiler to convert Cairo's intermediate representation "Sierra" code to machine code via MLIR and LLVM.

## Implemented Library Functions

Done:
1. `alloc_local`
1. `array_append`
1. `array_get`
1. `array_len`
1. `array_new`
1. `array_pop_front_consume`
1. `array_pop_front`
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
1. `u128_is_zero`
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

TODO:
1. `array_slice`
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
1. `u128_guarantee_mul`
1. `u128_mul_guarantee_verify`
1. `u512_safe_divmod_by_u256`


Footnotes
1. It is implemented but we're not sure if it has some stuff we don't know of.
2. It is implemented but we're still debating whether it should be a Rust-like `Box<T>` or if it's fine treating it like another variable.
3. It is implemented but side-effects are not yet handled (ex. array cloning/dropping).
4. Not supported by the Cairo to Sierra compiler.
5. Implemented with a dummy. It doesn't do anything yet.
6. It is implemented but we're not handling potential issues like lifetimes yet.

## Documentation

There is an mdbook in the `docs` folder. Build and read it with
```bash
make book
```

## Dependencies
- mdbook
- LLVM 16+ with MLIR
- Rust

## Setup

Install mdbook and other documentation dependencies:
```bash
cargo install mdbook mdbook-toc mdbook-mermaid
```

Install LLVM with MLIR. You can use the official packages provided by LLVM.

Install the cairo corelibs to be able to run the **tests** and compile `.cairo` programs to sierra:

```bash
./scripts/fetch-corelibs.sh
```

### Linux

Setup a environment variable called `MLIR_SYS_160_PREFIX` pointing to the llvm directory:

```bash
export MLIR_SYS_160_PREFIX=/usr/lib/llvm-16
```

### MacOS
```bash
brew install llvm@16
export MLIR_SYS_160_PREFIX=/opt/homebrew/opt/llvm@16
```

## CLI Interface

sierra2mlir:
```
Usage: sierra2mlir [OPTIONS] <INPUT>

Arguments:
  <INPUT>

Options:
  -o, --output <OUTPUT>  [default: -]
  -h, --help             Print help
```

sierrajit:
```
Usage: sierrajit [OPTIONS] <INPUT> <ENTRY_POINT>

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

## MLIR Resources
- https://mlir.llvm.org/docs/Tutorials/

## From MLIR to native binary
```bash
# to mlir with llvm dialect
cargo r --release --features build.cli --bin sierra2mlir -- program.sierra -o program.mlir

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
