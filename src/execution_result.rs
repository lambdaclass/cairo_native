use crate::{
    error::{jit_engine::ErrorImpl, JitRunnerError},
    invoke::JITValue,
    utils::u32_vec_to_felt,
    ExecuteResult,
};
use cairo_felt::Felt252;
use cairo_lang_sierra::extensions::core::CoreTypeConcrete;
use serde::{
    de::{self, SeqAccess},
    Deserializer,
};
use serde_json::Value;
use std::fmt::{self};

/// Starknet contract execution result.
#[derive(Debug, Default)]
pub struct NativeExecutionResult {
    pub remaining_gas: u128,
    pub failure_flag: bool,
    pub return_values: Vec<Felt252>,
    pub error_msg: Option<String>,
}

impl NativeExecutionResult {
    pub fn from_execute_result(result: ExecuteResult) -> Result<Self, JitRunnerError> {
        let mut error_msg = None;
        let mut failure_flag = false;

        assert_eq!(
            result.return_values.len(),
            1,
            "return values length doesnt match 1, which shouldn't happen with starknet contracts"
        );
        dbg!(&result);
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
                                            f.clone()
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
                                    f.clone()
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
            }
            _ => panic!("should be a enum"), // todo: handle as error
        };

        Ok(Self {
            remaining_gas: result.remaining_gas.unwrap_or(0),
            return_values,
            failure_flag,
            error_msg,
        })
    }
    /// Deserializes the NativeExecutionResult using the return types.
    ///
    /// The deserializer assumes a order in the incoming data, it expects the return values to be last (which is the case as of cairo 2).
    ///
    /// You can get the return types from a list of `CoreTypeConcrete` from a list of `ConcreteTypeId` using the sierra `ProgramRegistry`.
    pub fn deserialize_from_ret_types<'de, D>(
        deserializer: D,
        ret_types: &[&CoreTypeConcrete],
    ) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NativeExecutionResultVisitor<'a> {
            ret_types: &'a [&'a CoreTypeConcrete],
        }

        impl<'de> de::Visitor<'de> for NativeExecutionResultVisitor<'_> {
            type Value = NativeExecutionResult;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("Could not deserialize array of hexadecimal")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut remaining_gas: u128 = 0;

                for ret_type in self.ret_types {
                    match ret_type {
                        CoreTypeConcrete::GasBuiltin(_) => {
                            remaining_gas = seq.next_element()?.unwrap();
                        }
                        CoreTypeConcrete::RangeCheck(_) => {
                            seq.next_element::<Value>()?;
                        }
                        CoreTypeConcrete::Pedersen(_) => {
                            seq.next_element::<Value>()?;
                        }
                        CoreTypeConcrete::Poseidon(_) => {
                            seq.next_element::<Value>()?;
                        }
                        CoreTypeConcrete::StarkNet(_) => {
                            seq.next_element::<Value>()?;
                        }
                        CoreTypeConcrete::SegmentArena(_) => {
                            seq.next_element::<Value>()?;
                        }
                        CoreTypeConcrete::Enum(_) => {
                            // return values
                            // The failure flag indicates if the execution was done successfully.
                            let (failure_flag, return_values): (u64, Value) =
                                serde_json::from_value(seq.next_element::<Value>()?.unwrap())
                                    .unwrap();

                            return match failure_flag {
                                // When the execution is successful, the return values are
                                // stored in a nested vector. The innermost vector of u32
                                // represents a field element.
                                0 => {
                                    let return_values: Vec<Vec<Vec<Vec<u32>>>> =
                                        serde_json::from_value(return_values).unwrap();

                                    Ok(NativeExecutionResult {
                                        remaining_gas,
                                        return_values: return_values[0][0]
                                            .iter()
                                            .map(|felt_bytes| u32_vec_to_felt(felt_bytes))
                                            .collect(),
                                        failure_flag: failure_flag == 1,
                                        error_msg: None,
                                    })
                                }

                                // When the execution returns an error, the return values are
                                // a tuple with an empty array in the first place (don't really know
                                // why) and a vector of u32 vectors in the second. These represent a
                                // felt encoded string that gives some details about the error.
                                1 => {
                                    let return_values: (Vec<u32>, Vec<Vec<u32>>) =
                                        serde_json::from_value(return_values).unwrap();

                                    let felt_error: Vec<Felt252> = return_values
                                        .1
                                        .iter()
                                        .map(|felt_bytes| u32_vec_to_felt(felt_bytes))
                                        .collect();

                                    let str_error = String::from_utf8(
                                        felt_error
                                            .get(0)
                                            .ok_or_else(|| {
                                                de::Error::custom(
                                                    "error getting felt error message",
                                                )
                                            })?
                                            .to_be_bytes()
                                            .to_vec(),
                                    )
                                    .map_err(|_| {
                                        de::Error::custom("error parsing error from utf8")
                                    })?
                                    .trim_start_matches('\0')
                                    .to_owned();

                                    Ok(NativeExecutionResult {
                                        remaining_gas,
                                        failure_flag: failure_flag == 1,
                                        return_values: felt_error,
                                        error_msg: Some(str_error),
                                    })
                                }
                                _ => Err(de::Error::custom("expected failure flag to be 0 or 1")),
                            };
                        }
                        _ => Err(de::Error::custom("unexpected type when deserializing"))?,
                    }
                }

                Err(de::Error::custom("failed to deserialize"))
            }
        }

        const FIELDS: &[&str] = &["gas_consumed", "return_values", "failure_flag"];
        deserializer.deserialize_struct(
            "NativeExecutionResult",
            FIELDS,
            NativeExecutionResultVisitor { ret_types },
        )
    }
}
