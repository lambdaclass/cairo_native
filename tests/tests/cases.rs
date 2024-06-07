use crate::common::{
    compare_inputless_program, load_cairo_contract_path, run_native_starknet_contract,
    run_vm_contract,
};
use cairo_native::starknet::DummySyscallHandler;
use itertools::Itertools;
use num_traits::FromPrimitive;
use pretty_assertions_sorted::assert_eq_sorted;
use starknet_types_core::felt::Felt;
use test_case::test_case;

// Test cases for programs without input, it checks the outputs are correct automatically.

// felt tests
#[test_case("tests/cases/felt_ops/add.cairo")]
#[test_case("tests/cases/felt_ops/sub.cairo")]
#[test_case("tests/cases/felt_ops/felt_is_zero.cairo")]
#[test_case("tests/cases/felt_ops/mul.cairo")]
#[test_case("tests/cases/felt_ops/negation.cairo")]
#[test_case("tests/cases/felt_ops/div.cairo")]
// generic tests
#[test_case("tests/cases/fib_counter.cairo")]
#[test_case("tests/cases/fib_local.cairo")]
#[test_case("tests/cases/pedersen_hash.cairo")]
#[test_case("tests/cases/unwrap_non_zero.cairo")]
#[test_case("tests/cases/poseidon.cairo")]
#[test_case("tests/cases/panic_array.cairo")]
#[test_case("tests/cases/generic_fn_loop.cairo")]
// enums
#[test_case("tests/cases/enums/enum_init_c_style.cairo")]
#[test_case("tests/cases/enums/enum_init_empty.cairo")]
#[test_case("tests/cases/enums/enum_init_multiple.cairo")]
#[test_case("tests/cases/enums/enum_init_nested_c_style.cairo")]
#[test_case("tests/cases/enums/enum_init_nested_empty.cairo")]
#[test_case("tests/cases/enums/enum_init_nested_multiple.cairo")]
#[test_case("tests/cases/enums/enum_init_nested_single_scalar.cairo")]
#[test_case("tests/cases/enums/enum_init_nested_single_struct.cairo")]
#[test_case("tests/cases/enums/enum_init_nested_single_tuple.cairo")]
#[test_case("tests/cases/enums/enum_init_single_scalar.cairo")]
#[test_case("tests/cases/enums/enum_init_single_struct.cairo")]
#[test_case("tests/cases/enums/enum_init_single_tuple.cairo")]
#[test_case("tests/cases/enums/enum_init.cairo")]
#[test_case("tests/cases/enums/single_value.cairo")]
#[test_case("tests/cases/enums/enum_match.cairo")]
#[test_case("tests/cases/enums/enum_snapshot_match_a.cairo")]
#[test_case("tests/cases/enums/enum_snapshot_match_b.cairo")]
// returns
#[test_case("tests/cases/returns/enums.cairo")]
#[test_case("tests/cases/returns/simple.cairo")]
#[test_case("tests/cases/returns/tuple.cairo")]
// dict
#[test_case("tests/cases/dict/insert_get.cairo")]
// uint
#[test_case("tests/cases/uint/compare.cairo")]
#[test_case("tests/cases/uint/consts.cairo")]
#[test_case("tests/cases/uint/downcasts.cairo")]
#[test_case("tests/cases/uint/safe_divmod.cairo")]
#[test_case("tests/cases/uint/uint_addition.cairo")]
#[test_case("tests/cases/uint/uint_subtraction.cairo")]
#[test_case("tests/cases/uint/uint_try_from_felt.cairo")]
#[test_case("tests/cases/uint/upcasts.cairo")]
#[test_case("tests/cases/uint/wide_mul.cairo")]
#[test_case("tests/cases/uint/u512_safe_divmod_by_u256.cairo")]
// sint
#[test_case("tests/cases/sint/eq.cairo")]
#[test_case("tests/cases/sint/is_zero.cairo")]
#[test_case("tests/cases/sint/to_from_felt252.cairo")]
// sint8
#[test_case("tests/cases/sint/i8_diff.cairo")]
#[test_case("tests/cases/sint/i8_add_sub.cairo")]
#[test_case("tests/cases/sint/i8_wide_mul.cairo")]
// sint16
#[test_case("tests/cases/sint/i16_diff.cairo")]
#[test_case("tests/cases/sint/i16_add_sub.cairo")]
#[test_case("tests/cases/sint/i16_wide_mul.cairo")]
// sint32
#[test_case("tests/cases/sint/i32_diff.cairo")]
#[test_case("tests/cases/sint/i32_add_sub.cairo")]
#[test_case("tests/cases/sint/i32_wide_mul.cairo")]
// sint64
#[test_case("tests/cases/sint/i64_diff.cairo")]
#[test_case("tests/cases/sint/i64_add_sub.cairo")]
#[test_case("tests/cases/sint/i64_wide_mul.cairo")]
// sint128
#[test_case("tests/cases/sint/i128_diff.cairo")]
#[test_case("tests/cases/sint/i128_add_sub.cairo")]
// structs
#[test_case("tests/cases/structs/basic.cairo")]
#[test_case("tests/cases/structs/bigger.cairo")]
#[test_case("tests/cases/structs/enum_member.cairo")]
#[test_case("tests/cases/structs/nested.cairo")]
#[test_case("tests/cases/structs/struct_snapshot_deconstruct.cairo")]
// gas
#[test_case("tests/cases/gas/available_gas.cairo")]
// bool
#[test_case("tests/cases/bool/and.cairo")]
#[test_case("tests/cases/bool/eq.cairo")]
#[test_case("tests/cases/bool/not.cairo")]
#[test_case("tests/cases/bool/or.cairo")]
#[test_case("tests/cases/bool/to_felt252.cairo")]
#[test_case("tests/cases/bool/xor.cairo")]
// bitwise
#[test_case("tests/cases/bitwise/and.cairo")]
#[test_case("tests/cases/bitwise/or.cairo")]
#[test_case("tests/cases/bitwise/xor.cairo")]
// array
#[test_case("tests/cases/array/append.cairo")]
#[test_case("tests/cases/array/index_invalid.cairo")]
#[test_case("tests/cases/array/pop_front_invalid.cairo")]
#[test_case("tests/cases/array/pop_front_valid.cairo")]
#[test_case("tests/cases/array/slice.cairo")]
// nullable
#[test_case("tests/cases/nullable/test_nullable.cairo")]
// Programs copied from the cairo-vm
// https://github.com/lambdaclass/cairo-vm/tree/main/cairo_programs/cairo-1-programs
#[test_case("tests/cases/cairo_vm/programs/array_append.cairo")]
#[test_case("tests/cases/cairo_vm/programs/array_get.cairo")]
#[test_case("tests/cases/cairo_vm/programs/array_integer_tuple.cairo")]
#[test_case("tests/cases/cairo_vm/programs/bitwise.cairo")]
#[test_case("tests/cases/cairo_vm/programs/bytes31_ret.cairo")]
#[test_case("tests/cases/cairo_vm/programs/dict_with_struct.cairo")]
#[test_case("tests/cases/cairo_vm/programs/dictionaries.cairo")]
#[test_case("tests/cases/cairo_vm/programs/ecdsa_recover.cairo")]
#[test_case("tests/cases/cairo_vm/programs/enum_flow.cairo")]
#[test_case("tests/cases/cairo_vm/programs/enum_match.cairo")]
#[test_case("tests/cases/cairo_vm/programs/factorial.cairo")]
#[test_case("tests/cases/cairo_vm/programs/felt_dict.cairo")]
#[test_case("tests/cases/cairo_vm/programs/felt_dict_squash.cairo")]
#[test_case("tests/cases/cairo_vm/programs/felt_span.cairo")]
#[test_case("tests/cases/cairo_vm/programs/fibonacci.cairo")]
#[test_case("tests/cases/cairo_vm/programs/hello.cairo")]
#[test_case("tests/cases/cairo_vm/programs/my_rectangle.cairo")]
#[test_case("tests/cases/cairo_vm/programs/null_ret.cairo")]
#[test_case("tests/cases/cairo_vm/programs/nullable_box_vec.cairo")]
#[test_case("tests/cases/cairo_vm/programs/nullable_dict.cairo")]
#[test_case("tests/cases/cairo_vm/programs/ops.cairo")]
#[test_case("tests/cases/cairo_vm/programs/pedersen_example.cairo")]
#[test_case("tests/cases/cairo_vm/programs/poseidon.cairo")]
#[test_case("tests/cases/cairo_vm/programs/poseidon_pedersen.cairo")]
#[test_case("tests/cases/cairo_vm/programs/primitive_types2.cairo")]
#[test_case("tests/cases/cairo_vm/programs/print.cairo")]
#[test_case("tests/cases/cairo_vm/programs/recursion.cairo")]
#[test_case("tests/cases/cairo_vm/programs/sample.cairo")]
#[test_case("tests/cases/cairo_vm/programs/short_string.cairo")]
#[test_case("tests/cases/cairo_vm/programs/simple.cairo")]
#[test_case("tests/cases/cairo_vm/programs/simple_struct.cairo")]
#[test_case("tests/cases/cairo_vm/programs/struct_span_return.cairo")]
#[test_case("tests/cases/cairo_vm/programs/tensor_new.cairo")]
#[test_case("tests/cases/brainfuck.cairo")]
fn test_program_cases(program_path: &str) {
    compare_inputless_program(program_path)
}

// Contracts copied from the cairo-vm
// https://github.com/lambdaclass/cairo-vm/tree/main/cairo_programs/cairo-1-contracts
#[test_case("tests/cases/cairo_vm/contracts/alloc_segment.cairo", &[])]
#[test_case("tests/cases/cairo_vm/contracts/assert_le_find_small_arcs.cairo", &[])]
#[test_case("tests/cases/cairo_vm/contracts/dict_test.cairo", &[])]
#[test_case("tests/cases/cairo_vm/contracts/divmod.cairo", &[100, 10])]
#[test_case("tests/cases/cairo_vm/contracts/factorial.cairo", &[10])]
#[test_case("tests/cases/cairo_vm/contracts/felt252_dict_entry_init.cairo", &[])]
#[test_case("tests/cases/cairo_vm/contracts/felt252_dict_entry_update.cairo", &[])]
#[test_case("tests/cases/cairo_vm/contracts/felt_252_dict.cairo", &[])]
#[test_case("tests/cases/cairo_vm/contracts/fib.cairo", &[10, 10, 10])]
#[test_case("tests/cases/cairo_vm/contracts/get_segment_arena_index.cairo", &[])]
#[test_case("tests/cases/cairo_vm/contracts/init_squash_data.cairo", &[10])]
#[test_case("tests/cases/cairo_vm/contracts/linear_split.cairo", &[10])]
#[test_case("tests/cases/cairo_vm/contracts/should_skip_squash_loop.cairo", &[])]
#[test_case("tests/cases/cairo_vm/contracts/test_less_than.cairo", &[10])]
#[test_case("tests/cases/cairo_vm/contracts/u128_sqrt.cairo", &[100])]
#[test_case("tests/cases/cairo_vm/contracts/u16_sqrt.cairo", &[100])]
#[test_case("tests/cases/cairo_vm/contracts/u256_sqrt.cairo", &[100])]
#[test_case("tests/cases/cairo_vm/contracts/u32_sqrt.cairo", &[100])]
#[test_case("tests/cases/cairo_vm/contracts/u64_sqrt.cairo", &[100])]
#[test_case("tests/cases/cairo_vm/contracts/u8_sqrt.cairo", &[100])]
#[test_case("tests/cases/cairo_vm/contracts/uint512_div_mod.cairo", &[])]
#[test_case("tests/cases/cairo_vm/contracts/widemul128.cairo", &[100, 100])]
#[test_case("tests/cases/cairo_vm/contracts/field_sqrt.cairo", &[])]
#[test_case("tests/cases/cairo_vm/contracts/random_ec_point.cairo", &[])]
// #[test_case("tests/cases/cairo_vm/contracts/alloc_constant_size.cairo", &[10, 10, 10])]
fn test_contract_cases(program_path: &str, args: &[u128]) {
    let args = args
        .iter()
        .map(|&arg| Felt::from_u128(arg).unwrap())
        .collect_vec();

    let contract = load_cairo_contract_path(program_path);
    let entrypoint = contract
        .entry_points_by_type
        .external
        .first()
        .unwrap()
        .function_idx;
    let program = contract.extract_sierra_program().unwrap();

    let native_result =
        run_native_starknet_contract(&program, entrypoint, &args, DummySyscallHandler);
    assert!(!native_result.failure_flag);
    let native_output = native_result.return_values;

    let vm_output = run_vm_contract(&contract, entrypoint, &args);

    assert_eq_sorted!(vm_output, native_output);
}
