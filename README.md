# Cairo Native
[![test](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml/badge.svg)](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml)
[![mdbook](https://img.shields.io/badge/mdbook-link-blue)](https://lambdaclass.github.io/cairo_native/)

A compiler to convert Cairo's intermediate representation "Sierra" code to machine code via MLIR and LLVM.

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
- LLVM 16

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

# Tasks, TODOs, Roadmap

## Implemented Library Functions
- [x] `alloc_local`
- [x] `array_append`
- [x] `array_get`
- [x] `array_len`
- [x] `array_new`
- [x] `array_pop_front_consume`
- [x] `array_pop_front`
- [ ] `array_slice`
- [x] `array_snapshot_pop_back`
- [x] `array_snapshot_pop_front`
- [x] `bitwise`
- [x] `bool_and_impl`
- [x] `bool_not_impl`
- [x] `bool_or_impl`
- [x] `bool_to_felt252`
- [x] `bool_xor_impl`
- [x] `branch_align`
- [ ] `call_contract_syscall` (StarkNet)
- [ ] `class_hash_const` (StarkNet)
- [ ] `class_hash_to_felt252` (StarkNet)
- [ ] `class_hash_try_from_felt252` (StarkNet)
- [ ] `contract_address_const` (StarkNet)
- [ ] `contract_address_to_felt252` (StarkNet)
- [ ] `contract_address_try_from_felt252` (StarkNet)
- [ ] `deploy_syscall` (StarkNet)
- [x] `disable_ap_tracking`
- [ ] `downcast`
- [x] `drop` (3)
- [x] `dup` (3)
- [x] `ec_neg`
- [x] `ec_point_from_x_nz`
- [x] `ec_point_is_zero`
- [x] `ec_point_try_new_nz`
- [x] `ec_point_unwrap`
- [x] `ec_point_zero`
- [x] `ec_state_add_mul`
- [x] `ec_state_add`
- [x] `ec_state_init`
- [x] `ec_state_try_finalize_nz`
- [ ] `emit_event_syscall` (StarkNet)
- [x] `enable_ap_tracking`
- [x] `enum_init`
- [x] `enum_match`
- [ ] `enum_snapshot_match`
- [x] `felt252_add_const` (4)
- [x] `felt252_add`
- [x] `felt252_const`
- [x] `felt252_dict_entry_finalize`
- [x] `felt252_dict_entry_get`
- [x] `felt252_dict_new`
- [x] `felt252_dict_squash`
- [x] `felt252_div_const` (4)
- [x] `felt252_div` (4)
- [x] `felt252_is_zero`
- [x] `felt252_mul_const` (4)
- [x] `felt252_mul`
- [x] `felt252_sub_const` (4)
- [x] `felt252_sub`
- [x] `finalize_locals`
- [x] `function_call`
- [ ] `get_available_gas`
- [ ] `get_block_hash_syscall` (StarkNet)
- [x] `get_builtin_costs` (5)
- [ ] `get_execution_info_syscall` (StarkNet)
- [x] `into_box` (2)
- [x] `jump`
- [ ] `keccak_syscall` (StarkNet)
- [ ] `library_call_syscall` (StarkNet)
- [ ] `match_nullable`
- [ ] `null`
- [ ] `nullable_from_box`
- [x] `pedersen`
- [ ] `pop_log` (StarkNet, testing)
- [ ] `poseidon`
- [x] `print`
- [ ] `redeposit_gas`
- [x] `rename`
- [ ] `replace_class_syscall` (StarkNet)
- [x] `revoke_ap_tracking` (1)
- [ ] `secp256k1_add_syscall` (StarkNet)
- [ ] `secp256k1_get_point_from_x_syscall` (StarkNet)
- [ ] `secp256k1_get_xy_syscall` (StarkNet)
- [ ] `secp256k1_mul_syscall` (StarkNet)
- [ ] `secp256k1_new_syscall` (StarkNet)
- [ ] `secp256r1_add_syscall` (StarkNet)
- [ ] `secp256r1_get_point_from_x_syscall` (StarkNet)
- [ ] `secp256r1_get_xy_syscall` (StarkNet)
- [ ] `secp256r1_mul_syscall` (StarkNet)
- [ ] `secp256r1_new_syscall` (StarkNet)
- [ ] `send_message_to_l1_syscall` (StarkNet)
- [ ] `set_account_contract_address` (StarkNet, testing)
- [ ] `set_block_number` (StarkNet, testing)
- [ ] `set_block_timestamp` (StarkNet, testing)
- [ ] `set_caller_address` (StarkNet, testing)
- [ ] `set_chain_id` (StarkNet, testing)
- [ ] `set_contract_address` (StarkNet, testing)
- [ ] `set_max_fee` (StarkNet, testing)
- [ ] `set_nonce` (StarkNet, testing)
- [ ] `set_sequencer_address` (StarkNet, testing)
- [ ] `set_signature` (StarkNet, testing)
- [ ] `set_transaction_hash` (StarkNet, testing)
- [ ] `set_version` (StarkNet, testing)
- [x] `snapshot_take` (6)
- [x] `storage_address_from_base_and_offset` (StarkNet)
- [x] `storage_address_from_base` (StarkNet)
- [x] `storage_address_to_felt252` (StarkNet)
- [x] `storage_address_try_from_felt252` (StarkNet)
- [x] `storage_base_address_const` (StarkNet)
- [x] `storage_base_address_from_felt252` (StarkNet)
- [ ] `storage_read_syscall` (StarkNet)
- [ ] `storage_write_syscall` (StarkNet)
- [x] `store_local`
- [x] `store_temp`
- [x] `struct_construct`
- [x] `struct_deconstruct`
- [ ] `struct_snapshot_deconstruct`
- [x] `u128_byte_reverse`
- [x] `u128_const`
- [x] `u128_eq`
- [ ] `u128_guarantee_mul`
- [x] `u128_is_zero`
- [ ] `u128_mul_guarantee_verify`
- [x] `u128_overflowing_add`
- [x] `u128_overflowing_sub`
- [x] `u128_safe_divmod`
- [ ] `u128_sqrt`
- [x] `u128_to_felt252`
- [x] `u128s_from_felt252`
- [x] `u16_const`
- [x] `u16_eq`
- [x] `u16_is_zero`
- [x] `u16_overflowing_add`
- [x] `u16_overflowing_sub`
- [x] `u16_safe_divmod`
- [ ] `u16_sqrt`
- [x] `u16_to_felt252`
- [x] `u16_try_from_felt252`
- [ ] `u16_wide_mul`
- [ ] `u256_is_zero`
- [ ] `u256_safe_divmod`
- [ ] `u256_sqrt`
- [x] `u32_const`
- [x] `u32_eq`
- [x] `u32_is_zero`
- [x] `u32_overflowing_add`
- [x] `u32_overflowing_sub`
- [x] `u32_safe_divmod`
- [ ] `u32_sqrt`
- [x] `u32_to_felt252`
- [x] `u32_try_from_felt252`
- [ ] `u32_wide_mul`
- [ ] `u512_safe_divmod_by_u256`
- [x] `u64_const`
- [x] `u64_eq`
- [x] `u64_is_zero`
- [x] `u64_overflowing_add`
- [x] `u64_overflowing_sub`
- [x] `u64_safe_divmod`
- [ ] `u64_sqrt`
- [x] `u64_to_felt252`
- [x] `u64_try_from_felt252`
- [ ] `u64_wide_mul`
- [x] `u8_const`
- [x] `u8_eq`
- [x] `u8_is_zero`
- [x] `u8_overflowing_add`
- [x] `u8_overflowing_sub`
- [x] `u8_safe_divmod`
- [ ] `u8_sqrt`
- [x] `u8_to_felt252`
- [x] `u8_try_from_felt252`
- [ ] `u8_wide_mul`
- [x] `unbox` (2)
- [x] `unwrap_non_zero`
- [x] `upcast`
- [x] `withdraw_gas_all` (5)
- [x] `withdraw_gas` (5)

Footnotes
1. It is implemented but we're not sure if it has some stuff we don't know of.
2. It is implemented but we're still debating whether it should be a Rust-like `Box<T>` or if it's fine treating it like another variable.
3. It is implemented but side-effects are not yet handled (ex. array cloning/dropping).
4. Not supported by the Cairo to Sierra compiler.
5. Implemented with a dummy. It doesn't do anything yet.
6. It is implemented but we're not handling potential issues like lifetimes yet.
