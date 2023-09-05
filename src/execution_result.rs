use crate::utils::u32_vec_to_felt;
use cairo_felt::Felt252;
use serde::{
    de::{self, SeqAccess},
    Deserialize, Deserializer,
};
use serde_json::Value;
use std::fmt;

#[derive(Debug)]
pub struct NativeExecutionResult {
    pub gas_builtin: Option<u64>,
    pub range_check: Option<u64>,
    pub system: Option<u64>,
    pub failure_flag: bool,
    pub return_values: Vec<Felt252>,
    pub error_msg: Option<String>,
}

impl<'de> Deserialize<'de> for NativeExecutionResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NativeExecutionResultVisitor;

        impl<'de> de::Visitor<'de> for NativeExecutionResultVisitor {
            type Value = NativeExecutionResult;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("Could not deserialize array of hexadecimal")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                // The last element of the sequence is stored. This is where the
                // result of the MLIR execution will be.
                let mut last_element: Option<Value> = None;
                while let Some(value) = seq.next_element::<Option<Value>>()? {
                    last_element = value;
                }

                // The failure flag indicates if the execution was done successfully.
                let (failure_flag, return_values): (u64, Value) =
                    serde_json::from_value(last_element.unwrap()).unwrap();

                match failure_flag {
                    // When the execution is successful, the return values are
                    // stored in a nested vector. The innermost vector of u32
                    // represents a field element.
                    // TODO: This should be generalized for more return types
                    0 => {
                        let return_values: Vec<Vec<Vec<Vec<u32>>>> =
                            serde_json::from_value(return_values).unwrap();

                        return Ok(NativeExecutionResult {
                            gas_builtin: None,
                            range_check: None,
                            system: None,
                            return_values: return_values[0][0]
                                .iter()
                                .map(|felt_bytes| u32_vec_to_felt(felt_bytes))
                                .collect(),
                            failure_flag: failure_flag == 1,
                            error_msg: None,
                        });
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

                        let str_error = String::from_utf8(felt_error[0].to_be_bytes().to_vec())
                            .unwrap()
                            .trim_start_matches('\0')
                            .to_owned();

                        return Ok(NativeExecutionResult {
                            gas_builtin: None,
                            range_check: None,
                            system: None,
                            failure_flag: failure_flag == 1,
                            return_values: felt_error,
                            error_msg: Some(str_error),
                        });
                    }
                    _ => return Err(de::Error::custom("expected failure flag to be 0 or 1")),
                }
            }
        }

        const FIELDS: &'static [&'static str] = &[
            "gas_builtin",
            "range_check",
            "system",
            "return_values",
            "failure_flag",
        ];
        deserializer.deserialize_struct(
            "NativeExecutionResult",
            FIELDS,
            NativeExecutionResultVisitor,
        )
    }
}
