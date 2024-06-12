//use cairo_native::{
use cairo_native::{
//    execution_result::{BuiltinStats, ContractExecutionResult, ExecutionResult},
    execution_result::{BuiltinStats, ContractExecutionResult, ExecutionResult},
//    values::JitValue,
    values::JitValue,
//};
};
//use test_case::test_case;
use test_case::test_case;
//

//fn from_execution_result(
fn from_execution_result(
//    res: ExecutionResult,
    res: ExecutionResult,
//) -> Result<ContractExecutionResult, cairo_native::error::Error> {
) -> Result<ContractExecutionResult, cairo_native::error::Error> {
//    ContractExecutionResult::from_execution_result(res)
    ContractExecutionResult::from_execution_result(res)
//}
}
//

//#[test_case(
#[test_case(
//    JitValue::Enum {
    JitValue::Enum {
//        tag: 0,
        tag: 0,
//        value: Box::new(JitValue::Uint8(0)),
        value: Box::new(JitValue::Uint8(0)),
//        debug_name: None,
        debug_name: None,
//    } => panics "wrong type, expected: Struct { Struct { Array<felt252> } }, value: Uint8(0)")]
    } => panics "wrong type, expected: Struct { Struct { Array<felt252> } }, value: Uint8(0)")]
//#[test_case(
#[test_case(
//    JitValue::Enum {
    JitValue::Enum {
//        tag: 0,
        tag: 0,
//        value: Box::new(JitValue::Struct {
        value: Box::new(JitValue::Struct {
//            debug_name: None,
            debug_name: None,
//            fields: vec![JitValue::Uint8(0)]
            fields: vec![JitValue::Uint8(0)]
//        }),
        }),
//        debug_name: None,
        debug_name: None,
//    } => panics "wrong type, expected: Struct { Struct { Array<felt252> } }, value: Struct { fields: [Uint8(0)], debug_name: None }")]
    } => panics "wrong type, expected: Struct { Struct { Array<felt252> } }, value: Struct { fields: [Uint8(0)], debug_name: None }")]
//#[test_case(
#[test_case(
//    JitValue::Enum {
    JitValue::Enum {
//        tag: 0,
        tag: 0,
//        value: Box::new(JitValue::Struct {
        value: Box::new(JitValue::Struct {
//            debug_name: None,
            debug_name: None,
//            fields: vec![JitValue::Struct {
            fields: vec![JitValue::Struct {
//                debug_name: None,
                debug_name: None,
//                fields: vec![JitValue::Uint8(0)]
                fields: vec![JitValue::Uint8(0)]
//            }]
            }]
//        }),
        }),
//        debug_name: None,
        debug_name: None,
//    } => panics "wrong type, expected: Struct { Struct { Array<felt252> } }, value: Struct { fields: [Struct { fields: [Uint8(0)], debug_name: None }], debug_name: None }")]
    } => panics "wrong type, expected: Struct { Struct { Array<felt252> } }, value: Struct { fields: [Struct { fields: [Uint8(0)], debug_name: None }], debug_name: None }")]
//#[test_case(
#[test_case(
//    JitValue::Enum {
    JitValue::Enum {
//        tag: 0,
        tag: 0,
//        value: Box::new(JitValue::Struct {
        value: Box::new(JitValue::Struct {
//            debug_name: None,
            debug_name: None,
//            fields: vec![JitValue::Struct {
            fields: vec![JitValue::Struct {
//                debug_name: None,
                debug_name: None,
//                fields: vec![JitValue::Array(vec![JitValue::Uint8(0)])]
                fields: vec![JitValue::Array(vec![JitValue::Uint8(0)])]
//            }]
            }]
//        }),
        }),
//        debug_name: None,
        debug_name: None,
//    } => panics "should always be a felt")]
    } => panics "should always be a felt")]
//#[test_case(
#[test_case(
//    JitValue::Enum {
    JitValue::Enum {
//        tag: 1,
        tag: 1,
//        value: Box::new(JitValue::Uint8(0)),
        value: Box::new(JitValue::Uint8(0)),
//        debug_name: None,
        debug_name: None,
//    } => panics "wrong type, expected: Struct { [X, Array<felt252>] }, value: Uint8(0)")]
    } => panics "wrong type, expected: Struct { [X, Array<felt252>] }, value: Uint8(0)")]
//#[test_case(
#[test_case(
//    JitValue::Enum {
    JitValue::Enum {
//        tag: 1,
        tag: 1,
//        value: Box::new(JitValue::Struct {
        value: Box::new(JitValue::Struct {
//            debug_name: None,
            debug_name: None,
//            fields: vec![JitValue::Uint8(0)]
            fields: vec![JitValue::Uint8(0)]
//        }),
        }),
//        debug_name: None,
        debug_name: None,
//    } => panics "wrong type, expect: struct.fields.len() >= 2, value: [Uint8(0)]")]
    } => panics "wrong type, expect: struct.fields.len() >= 2, value: [Uint8(0)]")]
//#[test_case(
#[test_case(
//    JitValue::Enum {
    JitValue::Enum {
//        tag: 1,
        tag: 1,
//        value: Box::new(JitValue::Struct {
        value: Box::new(JitValue::Struct {
//            debug_name: None,
            debug_name: None,
//            fields: vec![JitValue::Uint8(0), JitValue::Uint8(0)]
            fields: vec![JitValue::Uint8(0), JitValue::Uint8(0)]
//        }),
        }),
//        debug_name: None,
        debug_name: None,
//    } => panics "wrong type, expected: Struct { [X, Array<felt252>] }, value: Struct { fields: [Uint8(0), Uint8(0)], debug_name: None }")]
    } => panics "wrong type, expected: Struct { [X, Array<felt252>] }, value: Struct { fields: [Uint8(0), Uint8(0)], debug_name: None }")]
//#[test_case(
#[test_case(
//    JitValue::Enum {
    JitValue::Enum {
//        tag: 1,
        tag: 1,
//        value: Box::new(JitValue::Struct {
        value: Box::new(JitValue::Struct {
//            debug_name: None,
            debug_name: None,
//            fields: vec![
            fields: vec![
//                JitValue::Uint8(0),
                JitValue::Uint8(0),
//                JitValue::Array(vec![JitValue::Uint8(0)])
                JitValue::Array(vec![JitValue::Uint8(0)])
//            ]
            ]
//        }),
        }),
//        debug_name: None,
        debug_name: None,
//    } => panics "should always be a felt")]
    } => panics "should always be a felt")]
//fn test_cases(return_value: JitValue) {
fn test_cases(return_value: JitValue) {
//    let _ = from_execution_result(ExecutionResult {
    let _ = from_execution_result(ExecutionResult {
//        return_value,
        return_value,
//        remaining_gas: None,
        remaining_gas: None,
//        builtin_stats: BuiltinStats::default(),
        builtin_stats: BuiltinStats::default(),
//    })
    })
//    .unwrap();
    .unwrap();
//}
}
