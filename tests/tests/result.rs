use cairo_native::{
    execution_result::{BuiltinStats, ContractExecutionResult, ExecutionResult},
    values::JitValue,
};
use test_case::test_case;

fn from_execution_result(
    res: ExecutionResult,
) -> Result<ContractExecutionResult, cairo_native::error::Error> {
    ContractExecutionResult::from_execution_result(res)
}

#[test_case(
    JitValue::Enum {
        tag: 0,
        value: Box::new(JitValue::Uint8(0)),
        debug_name: None,
    } => panics "wrong type, expect: outer struct, value: Uint8(0)")]
#[test_case(
    JitValue::Enum {
        tag: 0,
        value: Box::new(JitValue::Struct {
            debug_name: None,
            fields: vec![JitValue::Uint8(0)]
        }),
        debug_name: None,
    } => panics "wrong type, expect: inner struct, value: Uint8(0)")]
#[test_case(
    JitValue::Enum {
        tag: 0,
        value: Box::new(JitValue::Struct {
            debug_name: None,
            fields: vec![JitValue::Struct {
                debug_name: None,
                fields: vec![JitValue::Uint8(0)]
            }]
        }),
        debug_name: None,
    } => panics "wrong type, expect: array, value: Uint8(0)")]
#[test_case(
    JitValue::Enum {
        tag: 0,
        value: Box::new(JitValue::Struct {
            debug_name: None,
            fields: vec![JitValue::Struct {
                debug_name: None,
                fields: vec![JitValue::Array(vec![JitValue::Uint8(0)])]
            }]
        }),
        debug_name: None,
    } => panics "should always be a felt")]
#[test_case(
    JitValue::Enum {
        tag: 1,
        value: Box::new(JitValue::Uint8(0)),
        debug_name: None,
    } => panics "wrong type, expect: struct, value: Uint8(0)")]
#[test_case(
    JitValue::Enum {
        tag: 1,
        value: Box::new(JitValue::Struct {
            debug_name: None,
            fields: vec![JitValue::Uint8(0)]
        }),
        debug_name: None,
    } => panics "wrong type, expect: struct.fields.len() >= 2, value: [Uint8(0)]")]
#[test_case(
    JitValue::Enum {
        tag: 1,
        value: Box::new(JitValue::Struct {
            debug_name: None,
            fields: vec![JitValue::Uint8(0), JitValue::Uint8(0)]
        }),
        debug_name: None,
    } => panics "wrong type, expect: array, value: Uint8(0)")]
#[test_case(
    JitValue::Enum {
        tag: 1,
        value: Box::new(JitValue::Struct {
            debug_name: None,
            fields: vec![
                JitValue::Uint8(0),
                JitValue::Array(vec![JitValue::Uint8(0)])
            ]
        }),
        debug_name: None,
    } => panics "should always be a felt")]
fn test_cases(return_value: JitValue) {
    let _ = from_execution_result(ExecutionResult {
        return_value,
        remaining_gas: None,
        builtin_stats: BuiltinStats::default(),
    })
    .unwrap();
}
