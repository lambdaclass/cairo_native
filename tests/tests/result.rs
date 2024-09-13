use cairo_native::{
    execution_result::{BuiltinStats, ContractExecutionResult, ExecutionResult},
    Value,
};
use test_case::test_case;

fn from_execution_result(
    res: ExecutionResult,
) -> Result<ContractExecutionResult, cairo_native::error::Error> {
    ContractExecutionResult::from_execution_result(res)
}

#[test_case(
    Value::Enum {
        tag: 0,
        value: Box::new(Value::Uint8(0)),
        debug_name: None,
    } => panics "wrong type, expected: Struct { Struct { Array<felt252> } }, value: Uint8(0)")]
#[test_case(
    Value::Enum {
        tag: 0,
        value: Box::new(Value::Struct {
            debug_name: None,
            fields: vec![Value::Uint8(0)]
        }),
        debug_name: None,
    } => panics "wrong type, expected: Struct { Struct { Array<felt252> } }, value: Struct { fields: [Uint8(0)], debug_name: None }")]
#[test_case(
    Value::Enum {
        tag: 0,
        value: Box::new(Value::Struct {
            debug_name: None,
            fields: vec![Value::Struct {
                debug_name: None,
                fields: vec![Value::Uint8(0)]
            }]
        }),
        debug_name: None,
    } => panics "wrong type, expected: Struct { Struct { Array<felt252> } }, value: Struct { fields: [Struct { fields: [Uint8(0)], debug_name: None }], debug_name: None }")]
#[test_case(
    Value::Enum {
        tag: 0,
        value: Box::new(Value::Struct {
            debug_name: None,
            fields: vec![Value::Struct {
                debug_name: None,
                fields: vec![Value::Array(vec![Value::Uint8(0)])]
            }]
        }),
        debug_name: None,
    } => panics "should always be a felt")]
#[test_case(
    Value::Enum {
        tag: 1,
        value: Box::new(Value::Uint8(0)),
        debug_name: None,
    } => panics "wrong type, expected: Struct { [X, Array<felt252>] }, value: Uint8(0)")]
#[test_case(
    Value::Enum {
        tag: 1,
        value: Box::new(Value::Struct {
            debug_name: None,
            fields: vec![Value::Uint8(0)]
        }),
        debug_name: None,
    } => panics "wrong type, expect: struct.fields.len() >= 2, value: [Uint8(0)]")]
#[test_case(
    Value::Enum {
        tag: 1,
        value: Box::new(Value::Struct {
            debug_name: None,
            fields: vec![Value::Uint8(0), Value::Uint8(0)]
        }),
        debug_name: None,
    } => panics "wrong type, expected: Struct { [X, Array<felt252>] }, value: Struct { fields: [Uint8(0), Uint8(0)], debug_name: None }")]
#[test_case(
    Value::Enum {
        tag: 1,
        value: Box::new(Value::Struct {
            debug_name: None,
            fields: vec![
                Value::Uint8(0),
                Value::Array(vec![Value::Uint8(0)])
            ]
        }),
        debug_name: None,
    } => panics "should always be a felt")]
fn test_cases(return_value: Value) {
    let _ = from_execution_result(ExecutionResult {
        return_value,
        remaining_gas: None,
        builtin_stats: BuiltinStats::default(),
    })
    .unwrap();
}
