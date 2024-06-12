///// # Execution Result
/// # Execution Result
/////
///
///// This module contains the structures used to interpret the program execution results, either
/// This module contains the structures used to interpret the program execution results, either
///// normal programs or starknet contracts.
/// normal programs or starknet contracts.
//use crate::{error::Error, values::JitValue};
use crate::{error::Error, values::JitValue};
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//

//#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//pub struct BuiltinStats {
pub struct BuiltinStats {
//    pub bitwise: usize,
    pub bitwise: usize,
//    pub ec_op: usize,
    pub ec_op: usize,
//    pub range_check: usize,
    pub range_check: usize,
//    pub pedersen: usize,
    pub pedersen: usize,
//    pub poseidon: usize,
    pub poseidon: usize,
//    pub segment_arena: usize,
    pub segment_arena: usize,
//}
}
//

///// The result of the JIT execution.
/// The result of the JIT execution.
//#[derive(Debug, Clone, PartialEq, Eq)]
#[derive(Debug, Clone, PartialEq, Eq)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//pub struct ExecutionResult {
pub struct ExecutionResult {
//    pub remaining_gas: Option<u128>,
    pub remaining_gas: Option<u128>,
//    pub return_value: JitValue,
    pub return_value: JitValue,
//    pub builtin_stats: BuiltinStats,
    pub builtin_stats: BuiltinStats,
//}
}
//

///// Starknet contract execution result.
/// Starknet contract execution result.
//#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//pub struct ContractExecutionResult {
pub struct ContractExecutionResult {
//    pub remaining_gas: u128,
    pub remaining_gas: u128,
//    pub failure_flag: bool,
    pub failure_flag: bool,
//    pub return_values: Vec<Felt>,
    pub return_values: Vec<Felt>,
//    pub error_msg: Option<String>,
    pub error_msg: Option<String>,
//}
}
//

//impl ContractExecutionResult {
impl ContractExecutionResult {
//    /// Convert a [`ExecuteResult`] to a [`NativeExecutionResult`]
    /// Convert a [`ExecuteResult`] to a [`NativeExecutionResult`]
//    pub fn from_execution_result(result: ExecutionResult) -> Result<Self, Error> {
    pub fn from_execution_result(result: ExecutionResult) -> Result<Self, Error> {
//        let mut error_msg = None;
        let mut error_msg = None;
//        let failure_flag;
        let failure_flag;
//

//        let return_values = match &result.return_value {
        let return_values = match &result.return_value {
//            JitValue::Enum { tag, value, .. } => {
            JitValue::Enum { tag, value, .. } => {
//                failure_flag = *tag != 0;
                failure_flag = *tag != 0;
//

//                if !failure_flag {
                if !failure_flag {
//                    if let JitValue::Struct { fields, .. } = &**value {
                    if let JitValue::Struct { fields, .. } = &**value {
//                        if let JitValue::Struct { fields, .. } = &fields[0] {
                        if let JitValue::Struct { fields, .. } = &fields[0] {
//                            if let JitValue::Array(data) = &fields[0] {
                            if let JitValue::Array(data) = &fields[0] {
//                                let felt_vec: Vec<_> = data
                                let felt_vec: Vec<_> = data
//                                    .iter()
                                    .iter()
//                                    .map(|x| {
                                    .map(|x| {
//                                        if let JitValue::Felt252(f) = x {
                                        if let JitValue::Felt252(f) = x {
//                                            *f
                                            *f
//                                        } else {
                                        } else {
//                                            panic!("should always be a felt")
                                            panic!("should always be a felt")
//                                        }
                                        }
//                                    })
                                    })
//                                    .collect();
                                    .collect();
//                                felt_vec
                                felt_vec
//                            } else {
                            } else {
//                                Err(Error::UnexpectedValue(format!(
                                Err(Error::UnexpectedValue(format!(
//                                    "wrong type, expected: Struct {{ Struct {{ Array<felt252> }} }}, value: {:?}",
                                    "wrong type, expected: Struct {{ Struct {{ Array<felt252> }} }}, value: {:?}",
//                                    value
                                    value
//                                )))?
                                )))?
//                            }
                            }
//                        } else {
                        } else {
//                            Err(Error::UnexpectedValue(format!(
                            Err(Error::UnexpectedValue(format!(
//                                "wrong type, expected: Struct {{ Struct {{ Array<felt252> }} }}, value: {:?}",
                                "wrong type, expected: Struct {{ Struct {{ Array<felt252> }} }}, value: {:?}",
//                                value
                                value
//                            )))?
                            )))?
//                        }
                        }
//                    } else {
                    } else {
//                        Err(Error::UnexpectedValue(format!(
                        Err(Error::UnexpectedValue(format!(
//                            "wrong type, expected: Struct {{ Struct {{ Array<felt252> }} }}, value: {:?}",
                            "wrong type, expected: Struct {{ Struct {{ Array<felt252> }} }}, value: {:?}",
//                            value
                            value
//                        )))?
                        )))?
//                    }
                    }
//                } else if let JitValue::Struct { fields, .. } = &**value {
                } else if let JitValue::Struct { fields, .. } = &**value {
//                    if fields.len() < 2 {
                    if fields.len() < 2 {
//                        Err(Error::UnexpectedValue(format!(
                        Err(Error::UnexpectedValue(format!(
//                            "wrong type, expect: struct.fields.len() >= 2, value: {:?}",
                            "wrong type, expect: struct.fields.len() >= 2, value: {:?}",
//                            fields
                            fields
//                        )))?
                        )))?
//                    }
                    }
//                    if let JitValue::Array(data) = &fields[1] {
                    if let JitValue::Array(data) = &fields[1] {
//                        let felt_vec: Vec<_> = data
                        let felt_vec: Vec<_> = data
//                            .iter()
                            .iter()
//                            .map(|x| {
                            .map(|x| {
//                                if let JitValue::Felt252(f) = x {
                                if let JitValue::Felt252(f) = x {
//                                    *f
                                    *f
//                                } else {
                                } else {
//                                    panic!("should always be a felt")
                                    panic!("should always be a felt")
//                                }
                                }
//                            })
                            })
//                            .collect();
                            .collect();
//

//                        let bytes_err: Vec<_> = felt_vec
                        let bytes_err: Vec<_> = felt_vec
//                            .iter()
                            .iter()
//                            .flat_map(|felt| felt.to_bytes_be().to_vec())
                            .flat_map(|felt| felt.to_bytes_be().to_vec())
//                            // remove null chars
                            // remove null chars
//                            .filter(|b| *b != 0)
                            .filter(|b| *b != 0)
//                            .collect();
                            .collect();
//                        let str_error = String::from_utf8(bytes_err).unwrap().to_owned();
                        let str_error = String::from_utf8(bytes_err).unwrap().to_owned();
//

//                        error_msg = Some(str_error);
                        error_msg = Some(str_error);
//                        felt_vec
                        felt_vec
//                    } else {
                    } else {
//                        Err(Error::UnexpectedValue(format!(
                        Err(Error::UnexpectedValue(format!(
//                            "wrong type, expected: Struct {{ [X, Array<felt252>] }}, value: {:?}",
                            "wrong type, expected: Struct {{ [X, Array<felt252>] }}, value: {:?}",
//                            value
                            value
//                        )))?
                        )))?
//                    }
                    }
//                } else {
                } else {
//                    Err(Error::UnexpectedValue(format!(
                    Err(Error::UnexpectedValue(format!(
//                        "wrong type, expected: Struct {{ [X, Array<felt252>] }}, value: {:?}",
                        "wrong type, expected: Struct {{ [X, Array<felt252>] }}, value: {:?}",
//                        value
                        value
//                    )))?
                    )))?
//                }
                }
//            }
            }
//            _ => {
            _ => {
//                failure_flag = true;
                failure_flag = true;
//                Err(Error::UnexpectedValue(
                Err(Error::UnexpectedValue(
//                    "wrong return value type expected a enum".to_string(),
                    "wrong return value type expected a enum".to_string(),
//                ))?
                ))?
//            }
            }
//        };
        };
//

//        Ok(Self {
        Ok(Self {
//            remaining_gas: result.remaining_gas.unwrap_or(0),
            remaining_gas: result.remaining_gas.unwrap_or(0),
//            return_values,
            return_values,
//            failure_flag,
            failure_flag,
//            error_msg,
            error_msg,
//        })
        })
//    }
    }
//}
}
