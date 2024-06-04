use crate::common::compare_inputless_program;
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
#[test_case("tests/cases/cairo_vm/array_append.cairo")]
#[test_case("tests/cases/cairo_vm/array_get.cairo")]
#[test_case("tests/cases/cairo_vm/array_integer_tuple.cairo")]
#[test_case("tests/cases/cairo_vm/bitwise.cairo")]
#[test_case("tests/cases/cairo_vm/bytes31_ret.cairo")]
#[test_case("tests/cases/cairo_vm/dict_with_struct.cairo")]
#[test_case("tests/cases/cairo_vm/dictionaries.cairo")]
#[test_case("tests/cases/cairo_vm/ecdsa_recover.cairo")]
#[test_case("tests/cases/cairo_vm/enum_flow.cairo")]
#[test_case("tests/cases/cairo_vm/enum_match.cairo")]
#[test_case("tests/cases/cairo_vm/factorial.cairo")]
#[test_case("tests/cases/cairo_vm/felt_dict.cairo")]
#[test_case("tests/cases/cairo_vm/felt_dict_squash.cairo")]
#[test_case("tests/cases/cairo_vm/felt_span.cairo")]
#[test_case("tests/cases/cairo_vm/fibonacci.cairo")]
#[test_case("tests/cases/cairo_vm/hello.cairo")]
#[test_case("tests/cases/cairo_vm/my_rectangle.cairo")]
#[test_case("tests/cases/cairo_vm/null_ret.cairo")]
// #[test_case("tests/cases/cairo_vm/nullable_box_vec.cairo")]
#[test_case("tests/cases/cairo_vm/nullable_dict.cairo")]
#[test_case("tests/cases/cairo_vm/ops.cairo")]
#[test_case("tests/cases/cairo_vm/pedersen_example.cairo")]
#[test_case("tests/cases/cairo_vm/poseidon.cairo")]
#[test_case("tests/cases/cairo_vm/poseidon_pedersen.cairo")]
#[test_case("tests/cases/cairo_vm/primitive_types2.cairo")]
#[test_case("tests/cases/cairo_vm/print.cairo")]
#[test_case("tests/cases/cairo_vm/recursion.cairo")]
#[test_case("tests/cases/cairo_vm/sample.cairo")]
// #[test_case("tests/cases/cairo_vm/short_string.cairo")]
#[test_case("tests/cases/cairo_vm/simple.cairo")]
#[test_case("tests/cases/cairo_vm/simple_struct.cairo")]
#[test_case("tests/cases/cairo_vm/struct_span_return.cairo")]
#[test_case("tests/cases/cairo_vm/tensor_new.cairo")]
#[test_case("tests/cases/brainfuck.cairo")]
fn test_cases(program_path: &str) {
    compare_inputless_program(program_path)
}
