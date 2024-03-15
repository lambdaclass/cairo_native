use crate::{
    error::{jit_engine::ErrorImpl, JitRunnerError},
    values::JitValue,
};
use starknet_types_core::felt::Felt;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BuiltinStats {
    pub bitwise: usize,
    pub ec_op: usize,
    pub range_check: usize,
    pub pedersen: usize,
    pub poseidon: usize,
    pub segment_arena: usize,
}

/// The result of the JIT execution.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExecutionResult {
    pub remaining_gas: Option<u128>,
    pub return_value: JitValue,
    pub builtin_stats: BuiltinStats,
}

/// Starknet contract execution result.
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContractExecutionResult {
    pub remaining_gas: u128,
    pub failure_flag: bool,
    pub return_values: Vec<Felt>,
    pub error_msg: Option<String>,
}

impl ContractExecutionResult {
    /// Convert a [`ExecuteResult`] to a [`NativeExecutionResult`]
    pub fn from_execution_result(result: ExecutionResult) -> Result<Self, JitRunnerError> {
        let mut error_msg = None;
        let failure_flag;

        let return_values = match &result.return_value {
            JitValue::Enum { tag, value, .. } => {
                failure_flag = *tag != 0;

                if !failure_flag {
                    if let JitValue::Struct { fields, .. } = &**value {
                        if let JitValue::Struct { fields, .. } = &fields[0] {
                            if let JitValue::Array(data) = &fields[0] {
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
                                felt_vec
                            } else {
                                Err(JitRunnerError::from(ErrorImpl::UnexpectedValue(format!(
                                    "wrong type, expect: array, value: {:?}",
                                    value
                                ))))?
                            }
                        } else {
                            Err(JitRunnerError::from(ErrorImpl::UnexpectedValue(format!(
                                "wrong type, expect: struct, value: {:?}",
                                value
                            ))))?
                        }
                    } else {
                        Err(JitRunnerError::from(ErrorImpl::UnexpectedValue(format!(
                            "wrong type, expect: struct, value: {:?}",
                            value
                        ))))?
                    }
                } else if let JitValue::Struct { fields, .. } = &**value {
                    if let JitValue::Array(data) = &fields[1] {
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
                        Err(JitRunnerError::from(ErrorImpl::UnexpectedValue(format!(
                            "wrong type, expect: array, value: {:?}",
                            value
                        ))))?
                    }
                } else {
                    Err(JitRunnerError::from(ErrorImpl::UnexpectedValue(format!(
                        "wrong type, expect: struct, value: {:?}",
                        value
                    ))))?
                }
            }
            _ => {
                failure_flag = true;
                Err(JitRunnerError::from(ErrorImpl::UnexpectedValue(
                    "wrong return value type expected a enum".to_string(),
                )))?
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
