use crate::{
    error::{jit_engine::ErrorImpl, JitRunnerError},
    values::JITValue,
    ExecutionResult,
};
use starknet_types_core::felt::Felt;

/// Starknet contract execution result.
#[derive(Debug, Default)]
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

        assert_eq!(
            result.return_values.len(),
            1,
            "return values length doesnt match 1, which shouldn't happen with starknet contracts"
        );

        let return_values = match &result.return_values[0] {
            JITValue::Enum { tag, value, .. } => {
                failure_flag = *tag != 0;

                if !failure_flag {
                    if let JITValue::Struct { fields, .. } = &**value {
                        if let JITValue::Struct { fields, .. } = &fields[0] {
                            if let JITValue::Array(data) = &fields[0] {
                                let felt_vec: Vec<_> = data
                                    .iter()
                                    .map(|x| {
                                        if let JITValue::Felt252(f) = x {
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
                } else if let JITValue::Struct { fields, .. } = &**value {
                    if let JITValue::Array(data) = &fields[1] {
                        let felt_vec: Vec<_> = data
                            .iter()
                            .map(|x| {
                                if let JITValue::Felt252(f) = x {
                                    *f
                                } else {
                                    panic!("should always be a felt")
                                }
                            })
                            .collect();

                        let str_error =
                            String::from_utf8(felt_vec.get(0).unwrap().to_bytes_be().to_vec())
                                .unwrap()
                                .trim_start_matches('\0')
                                .to_owned();
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
