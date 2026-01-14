use crate::common::{compare_inputless_program, run_native_starknet_contract, run_vm_contract};
use cairo_native::{starknet::DummySyscallHandler, utils::testing::load_contract};
use itertools::Itertools;
use pretty_assertions_sorted::assert_eq_sorted;
use test_case::test_case;

// Test cases for programs without input, it checks the outputs are correct automatically.

// felt tests
#[test_case("cases/felt_ops/add")]
#[test_case("cases/felt_ops/sub")]
#[test_case("cases/felt_ops/felt_is_zero")]
#[test_case("cases/felt_ops/mul")]
#[test_case("cases/felt_ops/negation")]
#[test_case("cases/felt_ops/div")]
// generic tests
#[test_case("cases/fib_counter")]
#[test_case("cases/fib_local")]
#[test_case("cases/pedersen_hash")]
#[test_case("cases/unwrap_non_zero")]
#[test_case("cases/poseidon")]
#[test_case("cases/panic_array")]
#[test_case("cases/generic_fn_loop")]
// enums
#[test_case("cases/enums/enum_init_c_style")]
#[test_case("cases/enums/enum_init_empty")]
#[test_case("cases/enums/enum_init_multiple")]
#[test_case("cases/enums/enum_init_nested_c_style")]
#[test_case("cases/enums/enum_init_nested_empty")]
#[test_case("cases/enums/enum_init_nested_multiple")]
#[test_case("cases/enums/enum_init_nested_single_scalar")]
#[test_case("cases/enums/enum_init_nested_single_struct")]
#[test_case("cases/enums/enum_init_nested_single_tuple")]
#[test_case("cases/enums/enum_init_single_scalar")]
#[test_case("cases/enums/enum_init_single_struct")]
#[test_case("cases/enums/enum_init_single_tuple")]
#[test_case("cases/enums/enum_init")]
#[test_case("cases/enums/single_value")]
#[test_case("cases/enums/enum_match")]
#[test_case("cases/enums/enum_snapshot_match_a")]
#[test_case("cases/enums/enum_snapshot_match_b")]
// returns
#[test_case("cases/returns/enums")]
#[test_case("cases/returns/simple")]
#[test_case("cases/returns/tuple")]
// dict
#[test_case("cases/dict/insert_get")]
// uint
#[test_case("cases/uint/compare")]
#[test_case("cases/uint/consts")]
#[test_case("cases/uint/downcasts")]
#[test_case("cases/uint/safe_divmod")]
#[test_case("cases/uint/uint_addition")]
#[test_case("cases/uint/uint_subtraction")]
#[test_case("cases/uint/uint_try_from_felt")]
#[test_case("cases/uint/upcasts")]
#[test_case("cases/uint/wide_mul")]
#[test_case("cases/uint/u512_safe_divmod_by_u256")]
// sint
#[test_case("cases/sint/eq")]
#[test_case("cases/sint/to_from_felt252")]
// sint8
#[test_case("cases/sint/i8_diff")]
#[test_case("cases/sint/i8_add_sub")]
#[test_case("cases/sint/i8_wide_mul")]
// sint16
#[test_case("cases/sint/i16_diff")]
#[test_case("cases/sint/i16_add_sub")]
#[test_case("cases/sint/i16_wide_mul")]
// sint32
#[test_case("cases/sint/i32_diff")]
#[test_case("cases/sint/i32_add_sub")]
#[test_case("cases/sint/i32_wide_mul")]
// sint64
#[test_case("cases/sint/i64_diff")]
#[test_case("cases/sint/i64_add_sub")]
#[test_case("cases/sint/i64_wide_mul")]
// sint128
#[test_case("cases/sint/i128_diff")]
#[test_case("cases/sint/i128_add_sub")]
// structs
#[test_case("cases/structs/basic")]
#[test_case("cases/structs/bigger")]
#[test_case("cases/structs/enum_member")]
#[test_case("cases/structs/nested")]
#[test_case("cases/structs/struct_snapshot_deconstruct")]
// gas
#[test_case("cases/gas/available_gas")]
// bool
#[test_case("cases/bool/and")]
#[test_case("cases/bool/eq")]
#[test_case("cases/bool/not")]
#[test_case("cases/bool/or")]
#[test_case("cases/bool/to_felt252")]
#[test_case("cases/bool/xor")]
// bitwise
#[test_case("cases/bitwise/and")]
#[test_case("cases/bitwise/or")]
#[test_case("cases/bitwise/xor")]
// array
#[test_case("cases/array/append")]
#[test_case("cases/array/index_invalid")]
#[test_case("cases/array/pop_front_invalid")]
#[test_case("cases/array/pop_front_valid")]
#[test_case("cases/array/slice")]
// nullable
#[test_case("cases/nullable/test_nullable")]
// Programs copied from the cairo-vm
// https://github.com/lambdaclass/cairo-vm/tree/main/cairo_programs/cairo-1-programs
#[test_case("cases/cairo_vm/programs/array_append")]
#[test_case("cases/cairo_vm/programs/array_get")]
#[test_case("cases/cairo_vm/programs/array_integer_tuple")]
#[test_case("cases/cairo_vm/programs/bitwise")]
#[test_case("cases/cairo_vm/programs/bytes31_ret")]
#[test_case("cases/cairo_vm/programs/dict_with_struct")]
#[test_case("cases/cairo_vm/programs/dictionaries")]
#[test_case("cases/cairo_vm/programs/ecdsa_recover")]
#[test_case("cases/cairo_vm/programs/enum_flow")]
#[test_case("cases/cairo_vm/programs/enum_match")]
#[test_case("cases/cairo_vm/programs/factorial")]
#[test_case("cases/cairo_vm/programs/felt_dict")]
#[test_case("cases/cairo_vm/programs/felt_dict_squash")]
#[test_case("cases/cairo_vm/programs/felt_span")]
#[test_case("cases/cairo_vm/programs/fibonacci")]
#[test_case("cases/cairo_vm/programs/hello")]
#[test_case("cases/cairo_vm/programs/my_rectangle")]
#[test_case("cases/cairo_vm/programs/null_ret")]
#[test_case("cases/cairo_vm/programs/nullable_box_vec")]
#[test_case("cases/cairo_vm/programs/nullable_dict")]
#[test_case("cases/cairo_vm/programs/ops")]
#[test_case("cases/cairo_vm/programs/pedersen_example")]
#[test_case("cases/cairo_vm/programs/poseidon")]
#[test_case("cases/cairo_vm/programs/poseidon_pedersen")]
#[test_case("cases/cairo_vm/programs/primitive_types2")]
#[test_case("cases/cairo_vm/programs/print")]
#[test_case("cases/cairo_vm/programs/recursion")]
#[test_case("cases/cairo_vm/programs/sample")]
#[test_case("cases/cairo_vm/programs/short_string")]
#[test_case("cases/cairo_vm/programs/simple")]
#[test_case("cases/cairo_vm/programs/simple_struct")]
#[test_case("cases/cairo_vm/programs/struct_span_return")]
#[test_case("cases/cairo_vm/programs/tensor_new")]
#[test_case("cases/brainfuck")]
// EVM related
#[test_case("cases/stack")]
fn test_program_cases(program_path: &str) {
    compare_inputless_program(program_path)
}

// Contracts copied from the cairo-vm
// https://github.com/lambdaclass/cairo-vm/tree/main/cairo_programs/cairo-1-contracts
#[test_case("test_data_artifacts/contracts/cairo_vm/alloc_segment.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/assert_le_find_small_arcs.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/dict_test.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/divmod.contract.json", &[100, 10])]
#[test_case("test_data_artifacts/contracts/cairo_vm/factorial.contract.json", &[10])]
#[test_case("test_data_artifacts/contracts/cairo_vm/felt252_dict_entry_init.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/felt252_dict_entry_update.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/felt_252_dict.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/fib.contract.json", &[10, 10, 10])]
#[test_case("test_data_artifacts/contracts/cairo_vm/get_segment_arena_index.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/init_squash_data.contract.json", &[10])]
#[test_case("test_data_artifacts/contracts/cairo_vm/linear_split.contract.json", &[10])]
#[test_case("test_data_artifacts/contracts/cairo_vm/should_skip_squash_loop.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/test_less_than.contract.json", &[10])]
#[test_case("test_data_artifacts/contracts/cairo_vm/u128_sqrt.contract.json", &[100])]
#[test_case("test_data_artifacts/contracts/cairo_vm/u16_sqrt.contract.json", &[100])]
#[test_case("test_data_artifacts/contracts/cairo_vm/u256_sqrt.contract.json", &[100])]
#[test_case("test_data_artifacts/contracts/cairo_vm/u32_sqrt.contract.json", &[100])]
#[test_case("test_data_artifacts/contracts/cairo_vm/u64_sqrt.contract.json", &[100])]
#[test_case("test_data_artifacts/contracts/cairo_vm/u8_sqrt.contract.json", &[100])]
#[test_case("test_data_artifacts/contracts/cairo_vm/uint512_div_mod.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/widemul128.contract.json", &[100, 100])]
#[test_case("test_data_artifacts/contracts/cairo_vm/field_sqrt.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/random_ec_point.contract.json", &[])]
#[test_case("test_data_artifacts/contracts/cairo_vm/alloc_constant_size.contract.json", &[10, 10, 10])]
#[test_case("test_data_artifacts/contracts/cairo_vm/heavy_blake2s.contract.json", &[])]
fn test_contract_cases(name: &str, args: &[u128]) {
    let args = args.iter().map(|&arg| arg.into()).collect_vec();

    let contract = load_contract(name);

    let entrypoint = contract
        .entry_points_by_type
        .external
        .first()
        .expect("contract should have at least one external entrypoint");

    let program = contract
        .extract_sierra_program()
        .expect("contract bytes should be a valid sierra program");

    let native_result = run_native_starknet_contract(
        &program,
        entrypoint.function_idx,
        &args,
        DummySyscallHandler,
    );

    assert!(
        !native_result.failure_flag,
        "native contract execution failed"
    );

    let native_output = native_result.return_values;

    let vm_output = run_vm_contract(&contract, &entrypoint.selector, &args);

    assert_eq_sorted!(vm_output, native_output);
}
