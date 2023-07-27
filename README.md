# Cairo Native
[![test](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml/badge.svg)](https://github.com/lambdaclass/cairo_native/actions/workflows/ci.yml)
[![mdbook](https://img.shields.io/badge/mdbook-link-blue)](https://lambdaclass.github.io/cairo_native/)

A compiler to convert Cairo's intermediate representation "Sierra" code to machine code via MLIR and LLVM.

## Implemented Library Functions

Done:
1. `alloc_local`
2. `array_append`
3. `array_get`
4. `array_len`
5. `array_new`
6. `array_pop_front_consume`
7. `array_pop_front`
8. `array_snapshot_pop_back`
9. `array_snapshot_pop_front`
10. `bitwise`
11. `bool_and_impl`
12. `bool_not_impl`
13. `bool_or_impl`
14. `bool_to_felt252`
15. `bool_xor_impl`
16. `branch_align`
17. `disable_ap_tracking`
18. `downcast`
19. `drop` (3)
20. `dup` (3)
21. `ec_neg`
22. `ec_point_from_x_nz`
23. `ec_point_is_zero`
24. `ec_point_try_new_nz`
25. `ec_point_unwrap`
26. `ec_point_zero`
27. `ec_state_add_mul`
28. `ec_state_add`
29. `ec_state_init`
30. `ec_state_try_finalize_nz`
31. `enable_ap_tracking`
32. `enum_init`
33. `enum_match`
34. `felt252_add_const` (4)
35. `felt252_add`
36. `felt252_const`
37. `felt252_dict_entry_finalize`
38. `felt252_dict_entry_get`
39. `felt252_dict_new`
40. `felt252_dict_squash`
41. `felt252_div_const` (4)
42. `felt252_div` (4)
43. `felt252_is_zero`
44. `felt252_mul_const` (4)
45. `felt252_mul`
46. `felt252_sub_const` (4)
47. `felt252_sub`
48. `finalize_locals`
49. `function_call`
50. `get_block_hash_syscall` (StarkNet)
51. `get_builtin_costs` (5)
52. `into_box` (2)
53. `jump`
54. `match_nullable`
55. `null`
56. `nullable_from_box`
57. `pedersen`
58. `print`
59. `rename`
60. `revoke_ap_tracking` (1)
61. `snapshot_take` (6)
62. `storage_address_from_base_and_offset` (StarkNet)
63. `storage_address_from_base` (StarkNet)
64. `storage_address_to_felt252` (StarkNet)
65. `storage_address_try_from_felt252` (StarkNet)
66. `storage_base_address_const` (StarkNet)
67. `storage_base_address_from_felt252` (StarkNet)
68. `store_local`
69. `store_temp`
70. `struct_construct`
71. `struct_deconstruct`
72. `u128_byte_reverse`
73. `u128_const`
74. `u128_eq`
75. `u128_is_zero`
76. `u128_overflowing_add`
77. `u128_overflowing_sub`
78. `u128_safe_divmod`
79. `u128_sqrt`
80. `u128_to_felt252`
81. `u128s_from_felt252`
82. `u16_const`
83. `u16_eq`
84. `u16_is_zero`
85. `u16_overflowing_add`
86. `u16_overflowing_sub`
87. `u16_safe_divmod`
88. `u16_sqrt`
89. `u16_to_felt252`
90. `u16_try_from_felt252`
91. `u16_wide_mul`
92. `u32_const`
93. `u32_eq`
94. `u32_is_zero`
95. `u32_overflowing_add`
96. `u32_overflowing_sub`
97. `u32_safe_divmod`
98. `u32_sqrt`
99. `u32_to_felt252`
100. `u32_try_from_felt252`
101. `u32_wide_mul`
102. `u64_const`
103. `u64_eq`
104. `u64_is_zero`
105. `u64_overflowing_add`
106. `u64_overflowing_sub`
107. `u64_safe_divmod`
108. `u64_sqrt`
109. `u64_to_felt252`
110. `u64_try_from_felt252`
111. `u64_wide_mul`
112. `u8_const`
113. `u8_eq`
114. `u8_is_zero`
115. `u8_overflowing_add`
116. `u8_overflowing_sub`
117. `u8_safe_divmod`
118. `u8_sqrt`
119. `u8_to_felt252`
120. `u8_try_from_felt252`
121. `u8_wide_mul`
122. `unbox` (2)
123. `unwrap_non_zero`
124. `upcast`
125. `withdraw_gas_all` (5)
126. `withdraw_gas` (5)

TODO:
1. `array_slice`
2. `call_contract_syscall` (StarkNet)
3. `class_hash_const` (StarkNet)
4. `class_hash_to_felt252` (StarkNet)
5. `class_hash_try_from_felt252` (StarkNet)
6. `contract_address_const` (StarkNet)
7. `contract_address_to_felt252` (StarkNet)
8. `contract_address_try_from_felt252` (StarkNet)
9. `deploy_syscall` (StarkNet)
10. `emit_event_syscall` (StarkNet)
11. `enum_snapshot_match`
12. `get_available_gas`
13. `get_execution_info_syscall` (StarkNet)
14. `keccak_syscall` (StarkNet)
15. `library_call_syscall` (StarkNet)
16. `pop_log` (StarkNet, testing)
17. `poseidon`
18. `redeposit_gas`
19. `replace_class_syscall` (StarkNet)
20. `secp256k1_add_syscall` (StarkNet)
21. `secp256k1_get_point_from_x_syscall` (StarkNet)
22. `secp256k1_get_xy_syscall` (StarkNet)
23. `secp256k1_mul_syscall` (StarkNet)
24. `secp256k1_new_syscall` (StarkNet)
25. `secp256r1_add_syscall` (StarkNet)
26. `secp256r1_get_point_from_x_syscall` (StarkNet)
27. `secp256r1_get_xy_syscall` (StarkNet)
28. `secp256r1_mul_syscall` (StarkNet)
29. `secp256r1_new_syscall` (StarkNet)
30. `send_message_to_l1_syscall` (StarkNet)
31. `set_account_contract_address` (StarkNet, testing)
32. `set_block_number` (StarkNet, testing)
33. `set_block_timestamp` (StarkNet, testing)
34. `set_caller_address` (StarkNet, testing)
35. `set_chain_id` (StarkNet, testing)
36. `set_contract_address` (StarkNet, testing)
37. `set_max_fee` (StarkNet, testing)
38. `set_nonce` (StarkNet, testing)
39. `set_sequencer_address` (StarkNet, testing)
40. `set_signature` (StarkNet, testing)
41. `set_transaction_hash` (StarkNet, testing)
42. `set_version` (StarkNet, testing)
43. `storage_read_syscall` (StarkNet)
44. `storage_write_syscall` (StarkNet)
45. `struct_snapshot_deconstruct`
46. `u128_guarantee_mul`
47. `u128_mul_guarantee_verify`
48. `u256_is_zero`
49. `u256_safe_divmod`
50. `u256_sqrt`
51. `u512_safe_divmod_by_u256`


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
