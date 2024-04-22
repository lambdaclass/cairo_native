use cairo_native::{
    execution_result::{BuiltinStats, ContractExecutionResult, ExecutionResult},
    values::JitValue,
};

macro_rules! test_failure_execution_result {
    ($test:ident, $execution_result: expr, $expected_error: expr) => {
        #[test]
        #[should_panic(expected = $expected_error)]
        fn $test() {
            let _ = ContractExecutionResult::from_execution_result($execution_result).unwrap();
        }
    };
}

// Test that the error message is correct when the result
// doesn't contain an outer struct and tag is zero.
test_failure_execution_result!(
    test_execution_result_zero_tag_unexpected_result_struct_outer,
    ExecutionResult {
        return_value: JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Uint8(0)),
            debug_name: None,
        },
        remaining_gas: None,
        builtin_stats: BuiltinStats::default()
    },
    "wrong type, expect: outer struct, value: Uint8(0)"
);

// Test that the error message is correct when the result
// doesn't contain an inner struct and tag is zero.
test_failure_execution_result!(
    test_execution_result_zero_tag_unexpected_result_struct_inner,
    ExecutionResult {
        return_value: JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                debug_name: None,
                fields: vec![JitValue::Uint8(0)]
            }),
            debug_name: None,
        },
        remaining_gas: None,
        builtin_stats: BuiltinStats::default()
    },
    "wrong type, expect: inner struct, value: Uint8(0)"
);

// Test that the error message is correct when the result
// doesn't contain an array and tag is zero.
test_failure_execution_result!(
    test_execution_result_zero_tag_unexpected_result_array,
    ExecutionResult {
        return_value: JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                debug_name: None,
                fields: vec![JitValue::Struct {
                    debug_name: None,
                    fields: vec![JitValue::Uint8(0)]
                }]
            }),
            debug_name: None,
        },
        remaining_gas: None,
        builtin_stats: BuiltinStats::default()
    },
    "wrong type, expect: array, value: Uint8(0)"
);

// Test that the function panics when the result
// contains an array with non felt values and tag is zero.
test_failure_execution_result!(
    test_execution_result_zero_tag_unexpected_result_non_felt_array,
    ExecutionResult {
        return_value: JitValue::Enum {
            tag: 0,
            value: Box::new(JitValue::Struct {
                debug_name: None,
                fields: vec![JitValue::Struct {
                    debug_name: None,
                    fields: vec![JitValue::Array(vec![JitValue::Uint8(0)])]
                }]
            }),
            debug_name: None,
        },
        remaining_gas: None,
        builtin_stats: BuiltinStats::default()
    },
    "should always be a felt"
);

// Test that the test error message is correct when the result
// doesn't contain an outer struct and tag is one.
test_failure_execution_result!(
    test_execution_result_nonzero_tag_unexpected_result_struct,
    ExecutionResult {
        return_value: JitValue::Enum {
            tag: 1,
            value: Box::new(JitValue::Uint8(0)),
            debug_name: None,
        },
        remaining_gas: None,
        builtin_stats: BuiltinStats::default()
    },
    "wrong type, expect: struct, value: Uint8(0)"
);

// Test that the test error message is correct when the result
// is a struct with only one field and tag is one.
test_failure_execution_result!(
    test_execution_result_nonzero_tag_unexpected_result_struct_fields_len,
    ExecutionResult {
        return_value: JitValue::Enum {
            tag: 1,
            value: Box::new(JitValue::Struct {
                debug_name: None,
                fields: vec![JitValue::Uint8(0)]
            }),
            debug_name: None,
        },
        remaining_gas: None,
        builtin_stats: BuiltinStats::default()
    },
    "wrong type, expect: struct.fields.len() >= 2, value: [Uint8(0)]"
);

// Test that the test error message is correct when the result
// isn't an array and tag is one.
test_failure_execution_result!(
    test_execution_result_nonzero_tag_unexpected_result_array,
    ExecutionResult {
        return_value: JitValue::Enum {
            tag: 1,
            value: Box::new(JitValue::Struct {
                debug_name: None,
                fields: vec![JitValue::Uint8(0), JitValue::Uint8(0)]
            }),
            debug_name: None,
        },
        remaining_gas: None,
        builtin_stats: BuiltinStats::default()
    },
    "wrong type, expect: array, value: Uint8(0)"
);

// Test that the function panics when the result
// contains an array with non felt values and tag is one.
test_failure_execution_result!(
    test_execution_result_nonzero_tag_unexpected_result_non_felt_array,
    ExecutionResult {
        return_value: JitValue::Enum {
            tag: 1,
            value: Box::new(JitValue::Struct {
                debug_name: None,
                fields: vec![
                    JitValue::Uint8(0),
                    JitValue::Array(vec![JitValue::Uint8(0)])
                ]
            }),
            debug_name: None,
        },
        remaining_gas: None,
        builtin_stats: BuiltinStats::default()
    },
    "should always be a felt"
);
