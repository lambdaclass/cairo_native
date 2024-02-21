//! # Cairo native invocation results

use crate::{
    error::executor::{Error, ErrorImpl, Result},
    values::JitValue,
};
use core::fmt;
use starknet_types_core::felt::Felt;
use thiserror::Error;

/// Builtin usage statistics.
///
/// These statistics contain the number of times a libfunc invocation has been executed that used
/// (or rather, received a handle to) each of the builtins.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BuiltinStats {
    /// Number of usages of the `Bitwise` builtin.
    pub bitwise: usize,
    /// Number of usages of the `EcOp` builtin.
    pub ec_op: usize,
    /// Number of usages of the `RangeCheck` builtin.
    pub range_check: usize,
    /// Number of usages of the `Pedersen` builtin.
    pub pedersen: usize,
    /// Number of usages of the `Poseidon` builtin.
    pub poseidon: usize,
    /// Number of usages of the `SegmentArena` builtin.
    pub segment_arena: usize,
}

/// The result of the JIT execution.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExecutionResult {
    /// Amount of remaining gas.
    pub remaining_gas: Option<u128>,
    /// Returned value from the entry point.
    pub return_value: JitValue,
    /// Builtin usage statistics.
    pub builtin_stats: BuiltinStats,
}

/// Starknet contract execution result.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContractExecutionResult {
    /// Amount of remaining gas.
    pub remaining_gas: u128,
    /// The contract execution result.
    pub result: std::result::Result<Vec<Felt>, ContractExecutionError>,
}

/// Starknet contract execution error.
///
/// Internally, it's just a list of felts. Can be converted into a string.
#[derive(Debug, Clone, Error, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContractExecutionError(pub Vec<Felt>);

impl ContractExecutionError {
    fn rewrite_into_string(&self) -> String {
        self.0
            .iter()
            .map(|msg| {
                let mut buf = String::from_utf8(msg.to_bytes_be().to_vec())
                    .unwrap()
                    .trim_start_matches('\0')
                    .to_owned();
                buf.push('\n');
                buf
            })
            .collect()
    }
}

impl fmt::Display for ContractExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.rewrite_into_string())
    }
}

impl From<ContractExecutionError> for String {
    fn from(val: ContractExecutionError) -> Self {
        val.rewrite_into_string()
    }
}

impl ContractExecutionResult {
    /// Convert a [`ExecuteResult`] to a [`NativeExecutionResult`]
    pub fn from_execution_result(result: ExecutionResult) -> Result<Self> {
        let execution_result = match &result.return_value {
            JitValue::Enum { tag, value, .. } => {
                let failure_flag = *tag != 0;

                if !failure_flag {
                    if let JitValue::Struct { fields, .. } = &**value {
                        if let [JitValue::Struct { fields, .. }] = fields.as_slice() {
                            if let [JitValue::Array(data)] = fields.as_slice() {
                                let felt_vec: Vec<_> = data
                                    .iter()
                                    .map(|x| {
                                        if let JitValue::Felt252(f) = x {
                                            *f
                                        } else {
                                            panic!("should always be a felt")
                                        }
                                    })
                                    .collect();
                                Ok(felt_vec)
                            } else {
                                Err(Error::from(ErrorImpl::UnexpectedType(format!(
                                    "wrong type, expect: array, value: {:?}",
                                    value
                                ))))?
                            }
                        } else {
                            Err(Error::from(ErrorImpl::UnexpectedType(format!(
                                "wrong type, expect: struct, value: {:?}",
                                value
                            ))))?
                        }
                    } else {
                        Err(Error::from(ErrorImpl::UnexpectedType(format!(
                            "wrong type, expect: struct, value: {:?}",
                            value
                        ))))?
                    }
                } else if let JitValue::Struct { fields, .. } = &**value {
                    if let [_, JitValue::Array(data)] = fields.as_slice() {
                        Err(ContractExecutionError(
                            data.iter()
                                .map(|x| {
                                    if let JitValue::Felt252(f) = x {
                                        *f
                                    } else {
                                        panic!("should always be a felt")
                                    }
                                })
                                .collect(),
                        ))
                    } else {
                        Err(Error::from(ErrorImpl::UnexpectedType(format!(
                            "wrong type, expect: array, value: {:?}",
                            value
                        ))))?
                    }
                } else {
                    Err(Error::from(ErrorImpl::UnexpectedType(format!(
                        "wrong type, expect: struct, value: {:?}",
                        value
                    ))))?
                }
            }
            _ => Err(Error::from(ErrorImpl::UnexpectedType(
                "wrong return value type expected a enum".to_string(),
            )))?,
        };

        Ok(Self {
            remaining_gas: result.remaining_gas.unwrap_or(0),
            result: execution_result,
        })
    }
}
