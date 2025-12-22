use std::ops::{Add, AddAssign};

/// # Execution Result
///
/// This module contains the structures used to interpret the program execution results, either
/// normal programs or starknet contracts.
use crate::{error::Error, native_panic, utils::decode_error_message, values::Value};
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
    pub range_check: usize,
    pub pedersen: usize,
    pub bitwise: usize,
    pub ec_op: usize,
    pub poseidon: usize,
    pub segment_arena: usize,
    pub range_check96: usize,
    pub add_mod: usize,
    pub mul_mod: usize,
}

pub const RANGE_CHECK_BUILTIN_SIZE: usize = 1;
pub const PEDERSEN_BUILTIN_SIZE: usize = 3;
pub const BITWISE_BUILTIN_SIZE: usize = 5;
pub const EC_OP_BUILTIN_SIZE: usize = 7;
pub const POSEIDON_BUILTIN_SIZE: usize = 6;
pub const SEGMENT_ARENA_BUILTIN_SIZE: usize = 3;
pub const RANGE_CHECK96_BUILTIN_SIZE: usize = 1;
pub const ADD_MOD_BUILTIN_SIZE: usize = 7;
pub const MUL_MOD_BUILTIN_SIZE: usize = 7;

/// The result of the JIT execution.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionResult {
    pub remaining_gas: Option<u64>,
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
    pub remaining_gas: u64,
    pub failure_flag: bool,
    pub return_values: Vec<Felt>,
    pub error_msg: Option<String>,
    pub builtin_stats: BuiltinStats,
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
                                            Ok(*f)
                                        } else {
                                            native_panic!("should always be a felt")
                                        }
                                    })
                                    .collect::<Result<_, _>>()?;
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
                                    Ok(*f)
                                } else {
                                    native_panic!("should always be a felt")
                                }
                            })
                            .collect::<Result<_, _>>()?;

                        let bytes_err: Vec<_> = felt_vec
                            .iter()
                            .flat_map(|felt| felt.to_bytes_be().to_vec())
                            // remove null chars
                            .filter(|b| *b != 0)
                            .collect();
                        let str_error = decode_error_message(&bytes_err);

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
            builtin_stats: result.builtin_stats,
        })
    }
}

impl Add for BuiltinStats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            range_check: self.range_check + rhs.range_check,
            pedersen: self.pedersen + rhs.pedersen,
            bitwise: self.bitwise + rhs.bitwise,
            ec_op: self.ec_op + rhs.ec_op,
            poseidon: self.poseidon + rhs.poseidon,
            segment_arena: self.segment_arena + rhs.segment_arena,
            range_check96: self.range_check96 + rhs.range_check96,
            add_mod: self.add_mod + rhs.add_mod,
            mul_mod: self.mul_mod + rhs.mul_mod,
        }
    }
}

impl AddAssign for BuiltinStats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}
