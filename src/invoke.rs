//! A Rusty interface to provide parameters to JIT calls.

use cairo_felt::Felt252;
use serde::{ser::SerializeSeq, Serialize};

#[derive(Debug, Clone)]
pub enum InvokeArgs {
    Felt252(Felt252),
    Array(Vec<Self>),  // all elements need to be same type
    Struct(Vec<Self>), // element types can differ
    Enum { tag: u64, value: Box<Self> },
    Box(Box<Self>), // can't be null
    Nullable(Option<Box<Self>>),
    Uint8(u8),
    Uint16(u16),
    Uint32(u32),
    Uint64(u64),
    Uint128(u128),
}

#[derive(Debug, Clone, Default)]
pub struct InvokeContext {
    pub gas: Option<u128>,
    // Starknet syscall handler
    pub system: Option<u64>,
    pub bitwise: bool,
    pub range_check: bool,
    pub pedersen: bool,
    // call args
    pub args: Vec<InvokeArgs>,
}

impl Serialize for InvokeContext {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut params_seq = serializer.serialize_seq(None)?;

        // TODO: check order
        if self.bitwise {
            params_seq.serialize_element(&())?;
        }
        if self.range_check {
            params_seq.serialize_element(&())?;
        }
        if self.pedersen {
            params_seq.serialize_element(&())?;
        }

        if let Some(system) = self.system {
            params_seq.serialize_element(&system)?;
        }

        if let Some(gas) = self.gas {
            params_seq.serialize_element(&gas)?;
        }

        for arg in &self.args {
            params_seq.serialize_element(arg)?;
        }

        params_seq.end()
    }
}

impl Serialize for InvokeArgs {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            InvokeArgs::Felt252(value) => {
                serializer.serialize_bytes(value.to_be_bytes().as_slice())
            }
            InvokeArgs::Array(value) => {
                let mut seq = serializer.serialize_seq(Some(value.len()))?;
                for val in value {
                    seq.serialize_element(val)?;
                }
                seq.end()
            }
            InvokeArgs::Struct(value) => {
                let mut seq = serializer.serialize_seq(Some(value.len()))?;
                for val in value {
                    seq.serialize_element(val)?;
                }
                seq.end()
            }
            InvokeArgs::Enum { tag, value } => {
                let mut seq = serializer.serialize_seq(Some(2))?;
                seq.serialize_element(tag)?;
                seq.serialize_element(value.as_ref())?;
                seq.end()
            }
            InvokeArgs::Box(value) => serializer.serialize_some(value.as_ref()),
            InvokeArgs::Nullable(value) => match value {
                Some(value) => serializer.serialize_some(value.as_ref()),
                None => serializer.serialize_none(),
            },
            InvokeArgs::Uint8(value) => serializer.serialize_u8(*value),
            InvokeArgs::Uint16(value) => serializer.serialize_u16(*value),
            InvokeArgs::Uint32(value) => serializer.serialize_u32(*value),
            InvokeArgs::Uint64(value) => serializer.serialize_u64(*value),
            InvokeArgs::Uint128(value) => serializer.serialize_u128(*value),
        }
    }
}
