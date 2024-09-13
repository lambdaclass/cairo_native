/// # Execution Result
///
/// This module contains the structures used to interpret the program execution results, either
/// normal programs or starknet contracts.
use crate::{error::Error, values::Value};
use starknet_types_core::felt::Felt;

#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct BuiltinStats {
    pub bitwise: usize,
    pub ec_op: usize,
    pub range_check: usize,
    pub pedersen: usize,
    pub poseidon: usize,
    pub segment_arena: usize,
    pub range_check_96: usize,
    pub circuit_add: usize,
    pub circuit_mul: usize,
}

/// The result of the JIT execution.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionResult {
    pub remaining_gas: Option<u128>,
    pub return_value: Value,
    pub builtin_stats: BuiltinStats,
}

/// Starknet contract execution result.
#[derive(
    Debug,
    Default,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct ContractExecutionResult {
    pub remaining_gas: u128,
    pub failure_flag: bool,
    pub return_values: Vec<Felt>,
    pub error_msg: Option<String>,
}

impl ContractExecutionResult {
    /// Convert an [`ExecutionResult`] into a [`ContractExecutionResult`]
    pub fn from_execution_result(result: ExecutionResult) -> Result<Self, Error> {
        let mut error_msg = None;
        let failure_flag;

        let return_values = match &result.return_value {
            Value::Enum { tag, value, .. } => {
                failure_flag = *tag != 0;

                if !failure_flag {
                    if let Value::Struct { fields, .. } = &**value {
                        if let Value::Struct { fields, .. } = &fields[0] {
                            if let Value::Array(data) = &fields[0] {
                                let felt_vec: Vec<_> = data
                                    .iter()
                                    .map(|x| {
                                        if let Value::Felt252(f) = x {
                                            *f
                                        } else {
                                            panic!("should always be a felt")
                                        }
                                    })
                                    .collect();
                                felt_vec
                            } else {
                                Err(Error::UnexpectedValue(format!(
                                    "wrong type, expected: Struct {{ Struct {{ Array<felt252> }} }}, value: {:?}",
                                    value
                                )))?
                            }
                        } else {
                            Err(Error::UnexpectedValue(format!(
                                "wrong type, expected: Struct {{ Struct {{ Array<felt252> }} }}, value: {:?}",
                                value
                            )))?
                        }
                    } else {
                        Err(Error::UnexpectedValue(format!(
                            "wrong type, expected: Struct {{ Struct {{ Array<felt252> }} }}, value: {:?}",
                            value
                        )))?
                    }
                } else if let Value::Struct { fields, .. } = &**value {
                    if fields.len() < 2 {
                        Err(Error::UnexpectedValue(format!(
                            "wrong type, expect: struct.fields.len() >= 2, value: {:?}",
                            fields
                        )))?
                    }
                    if let Value::Array(data) = &fields[1] {
                        let felt_vec: Vec<_> = data
                            .iter()
                            .map(|x| {
                                if let Value::Felt252(f) = x {
                                    *f
                                } else {
                                    panic!("should always be a felt")
                                }
                            })
                            .collect();

                        let bytes_err: Vec<_> = felt_vec
                            .iter()
                            .flat_map(|felt| felt.to_bytes_be().to_vec())
                            // remove null chars
                            .filter(|b| *b != 0)
                            .collect();
                        let str_error = String::from_utf8(bytes_err).unwrap().to_owned();

                        error_msg = Some(str_error);
                        felt_vec
                    } else {
                        Err(Error::UnexpectedValue(format!(
                            "wrong type, expected: Struct {{ [X, Array<felt252>] }}, value: {:?}",
                            value
                        )))?
                    }
                } else {
                    Err(Error::UnexpectedValue(format!(
                        "wrong type, expected: Struct {{ [X, Array<felt252>] }}, value: {:?}",
                        value
                    )))?
                }
            }
            _ => {
                failure_flag = true;
                Err(Error::UnexpectedValue(
                    "wrong return value type expected a enum".to_string(),
                ))?
            }
        };

        Ok(Self {
            remaining_gas: result.remaining_gas.unwrap_or(0),
            return_values,
            failure_flag,
            error_msg,
        })
    }
}
