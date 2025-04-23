use crate::value::Value;
use cairo_lang_sierra::{ids::VarId, program::StatementIdx};
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
use serde::{ser::SerializeMap, Serialize};
use starknet_crypto::Felt;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Serialize)]
pub struct ProgramTrace {
    pub states: Vec<StateDump>,
    // TODO: Syscall data.
}

impl ProgramTrace {
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    pub fn push(&mut self, state: StateDump) {
        self.states.push(state);
    }
}

#[derive(Clone, Debug)]
pub struct StateDump {
    pub statement_idx: StatementIdx,
    pub items: BTreeMap<u64, Value>,
}

impl StateDump {
    pub fn new(statement_idx: StatementIdx, state: OrderedHashMap<VarId, Value>) -> Self {
        Self {
            statement_idx,
            items: state
                .into_iter()
                .map(|(id, value)| (id.id, value))
                .collect(),
        }
    }
}

impl Serialize for StateDump {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut s = s.serialize_map(Some(2))?;

        s.serialize_entry("statementIdx", &self.statement_idx.0)?;
        s.serialize_entry("preStateDump", &self.items)?;

        s.end()
    }
}

#[derive(Debug, Clone)]
pub struct ContractExecutionResult {
    pub remaining_gas: u64,
    pub failure_flag: bool,
    pub return_values: Vec<Felt>,
    pub error_msg: Option<String>,
}

impl ContractExecutionResult {
    pub fn from_trace(trace: &ProgramTrace) -> Option<Self> {
        let last = trace.states.last()?;
        Self::from_state(last)
    }

    pub fn from_state(state: &StateDump) -> Option<Self> {
        let mut remaining_gas = None;
        let mut error_msg = None;
        let mut failure_flag = false;
        let mut return_values = Vec::new();

        for value in state.items.values() {
            match value {
                Value::U64(gas) => remaining_gas = Some(*gas),
                Value::Enum {
                    self_ty: _,
                    index,
                    payload,
                } => {
                    failure_flag = (*index) != 0;

                    if let Value::Struct(inner) = &**payload {
                        if !failure_flag {
                            if let Value::Struct(inner) = &inner[0] {
                                if let Value::Array { ty: _, data } = &inner[0] {
                                    for value in data.iter() {
                                        if let Value::Felt(x) = value {
                                            return_values.push(*x);
                                        }
                                    }
                                }
                            }
                        } else if let Value::Array { ty: _, data } = &inner[1] {
                            let mut error_felt_vec = Vec::new();
                            for value in data.iter() {
                                if let Value::Felt(x) = value {
                                    error_felt_vec.push(*x);
                                }
                            }
                            let bytes_err: Vec<_> = error_felt_vec
                                .iter()
                                .flat_map(|felt| felt.to_bytes_be().to_vec())
                                // remove null chars
                                .filter(|b| *b != 0)
                                .collect();
                            let str_error = String::from_utf8(bytes_err).unwrap().to_owned();
                            error_msg = Some(str_error);
                        }
                    }
                }
                Value::Unit => {}
                _ => None?,
            }
        }

        Some(Self {
            remaining_gas: remaining_gas.unwrap_or(0),
            return_values,
            error_msg,
            failure_flag,
        })
    }
}
