/// # Execution Result
///
/// This module contains the structures used to interpret the program execution results, either
/// normal programs or starknet contracts.
use crate::{error::Error, values::JitValue};
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
    pub fn from_execution_result(result: ExecutionResult) -> Result<Self, Error> {
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
                } else if let JitValue::Struct { fields, .. } = &**value {
                    if fields.len() < 2 {
                        Err(Error::UnexpectedValue(format!(
                            "wrong type, expect: struct.fields.len() >= 2, value: {:?}",
                            fields
                        )))?
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "with-serde")]
    use serde_json;

    #[test]
    #[cfg(feature = "with-serde")]
    fn test_builtin_stats_serialization_deserialization() {
        // Create an example of BuiltinStats
        let original_stats = BuiltinStats {
            bitwise: 10,
            ec_op: 20,
            range_check: 30,
            pedersen: 40,
            poseidon: 50,
            segment_arena: 60,
        };

        // Serialize to JSON
        let serialized = serde_json::to_string(&original_stats).expect("Failed to serialize");

        // Deserialize from JSON
        let deserialized: BuiltinStats =
            serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Verify that the deserialized result is equal to the original
        assert_eq!(original_stats, deserialized);
    }

    #[test]
    #[cfg(feature = "with-serde")]
    fn test_execution_result_serialization_deserialization() {
        // Create an example of ExecutionResult with various JitValue variants
        let original_result = ExecutionResult {
            remaining_gas: Some(1000),
            return_value: JitValue::Struct {
                fields: vec![
                    JitValue::Felt252(Felt::from(1234)),
                    JitValue::Uint64(42),
                    JitValue::Array(vec![
                        JitValue::Uint8(1),
                        JitValue::Uint8(2),
                        JitValue::Uint8(3),
                    ]),
                ],
                debug_name: Some("example_struct".to_string()),
            },
            builtin_stats: BuiltinStats {
                bitwise: 10,
                ec_op: 20,
                range_check: 30,
                pedersen: 40,
                poseidon: 50,
                segment_arena: 60,
            },
        };

        // Serialize to JSON
        let serialized = serde_json::to_string(&original_result).expect("Failed to serialize");

        // Deserialize from JSON
        let deserialized: ExecutionResult =
            serde_json::from_str(&serialized).expect("Failed to deserialize");

        // Verify that the deserialized result is equal to the original
        assert_eq!(original_result, deserialized);
    }

    #[test]
    #[cfg(feature = "with-serde")]
    fn test_contract_execution_result_serialization_deserialization() {
        // Create an example of ContractExecutionResult
        let original_result = ContractExecutionResult {
            remaining_gas: 1000,
            failure_flag: false,
            return_values: vec![Felt::from(1234), Felt::from(5678)],
            error_msg: Some("No error".to_string()),
        };

        // Serialize to JSON
        let serialized = serde_json::to_string(&original_result).expect("Failed to serialize");

        // Deserialize from JSON
        let deserialized: ContractExecutionResult =
            serde_json::from_str(&serialized).expect("Failed to deserialize");

        println!("deserialized: {:?}", deserialized);

        // Verify that the deserialized result is equal to the original
        assert_eq!(original_result, deserialized);
    }

    #[test]
    fn test_contract_execution_result_default() {
        // Create a default instance of ContractExecutionResult
        let default_result = ContractExecutionResult::default();

        // Verify that the default values are correct
        assert_eq!(
            default_result,
            ContractExecutionResult {
                remaining_gas: 0,
                failure_flag: false,
                return_values: Vec::new(),
                error_msg: None,
            }
        );
    }

    #[test]
    fn test_contract_execution_result_ordering() {
        // Create instances of ContractExecutionResult for comparison
        let result1 = ContractExecutionResult {
            remaining_gas: 1000,
            failure_flag: false,
            return_values: vec![Felt::from(1234)],
            error_msg: Some("No error".to_string()),
        };

        let result2 = ContractExecutionResult {
            remaining_gas: 2000,
            failure_flag: false,
            return_values: vec![Felt::from(1234)],
            error_msg: Some("No error".to_string()),
        };

        let result3 = ContractExecutionResult {
            remaining_gas: 1000,
            failure_flag: true,
            return_values: vec![Felt::from(1234)],
            error_msg: Some("Error".to_string()),
        };

        // Verify ordering
        assert!(result1 < result2);
        assert!(result2 > result1);
        assert!(result1 < result3);
        assert!(result3 > result1);
        assert!(result1 == result1.clone());
    }

    #[test]
    #[should_panic(expected = "wrong return value type expected a enum")]
    fn test_from_execution_result_non_enum() {
        // Create an ExecutionResult with a return_value that is not a JitValue::Enum
        let execution_result = ExecutionResult {
            remaining_gas: Some(1000),
            return_value: JitValue::Felt252(Felt::from(1234)), // Not an Enum
            builtin_stats: BuiltinStats {
                bitwise: 10,
                ec_op: 20,
                range_check: 30,
                pedersen: 40,
                poseidon: 50,
                segment_arena: 60,
            },
        };

        // Attempt to convert it to a ContractExecutionResult, expecting a panic
        ContractExecutionResult::from_execution_result(execution_result).unwrap();
    }
}
