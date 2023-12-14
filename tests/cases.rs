use common::compare_inputless_program;
use test_case::test_case;

mod common;

// Test cases for programs without input, it checks the outputs are correct automatically.

// felt tests
#[test_case("tests/cases/felt_ops/add.cairo")]
#[test_case("tests/cases/felt_ops/sub.cairo")]
#[test_case("tests/cases/felt_ops/felt_is_zero.cairo")]
#[test_case("tests/cases/felt_ops/mul.cairo")]
#[test_case("tests/cases/felt_ops/negation.cairo")]
#[test_case("tests/cases/felt_ops/div.cairo"  => ignore["not implemented yet"])]
// generic tests
#[test_case("tests/cases/fib_counter.cairo")]
#[test_case("tests/cases/fib_local.cairo")]
#[test_case("tests/cases/pedersen_hash.cairo")]
#[test_case("tests/cases/unwrap_non_zero.cairo")]
#[test_case("tests/cases/poseidon.cairo")]
// enums
// TODO: compare error: Fail(Reason("assertion failed: `(left == right)` \n  left: `0`,\n right: `10` at tests/common.rs:453"))
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
#[test_case("tests/cases/dict/insert_get.cairo" => ignore["gas mismatch"])]
// uint
#[test_case("tests/cases/uint/compare.cairo")]
#[test_case("tests/cases/uint/consts.cairo")]
#[test_case("tests/cases/uint/downcasts.cairo")]
#[test_case("tests/cases/uint/safe_divmod.cairo" => ignore["TODO: cairo program is outdated"])]
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
#[test_case("tests/cases/gas/available_gas.cairo" => ignore["unimplemented"])]
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
#[test_case("tests/cases/nullable/test_nullable.cairo" => ignore["unimplemented"])]
fn test_cases(program_path: &str) {
    compare_inputless_program(program_path)
}
