//! A Rusty interface to provide parameters to JIT calls.

use cairo_felt::Felt252;
use serde::{ser::SerializeSeq, Serialize};

use crate::{metadata::syscall_handler::SyscallHandlerMeta, utils::felt252_bigint};

#[derive(Debug, Clone)]
pub enum InvokeArg {
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

#[derive(Debug, Default)]
pub struct InvokeContext<'s> {
    pub gas: Option<u128>,
    // Starknet syscall handler
    pub system: Option<&'s SyscallHandlerMeta>,
    pub bitwise: bool,
    pub range_check: bool,
    pub pedersen: bool,
    // call args
    pub args: Vec<InvokeArg>,
}

// Conversions

impl From<Felt252> for InvokeArg {
    fn from(value: Felt252) -> Self {
        InvokeArg::Felt252(value)
    }
}

impl From<u8> for InvokeArg {
    fn from(value: u8) -> Self {
        InvokeArg::Uint8(value)
    }
}

impl From<u16> for InvokeArg {
    fn from(value: u16) -> Self {
        InvokeArg::Uint16(value)
    }
}

impl From<u32> for InvokeArg {
    fn from(value: u32) -> Self {
        InvokeArg::Uint32(value)
    }
}

impl From<u64> for InvokeArg {
    fn from(value: u64) -> Self {
        InvokeArg::Uint64(value)
    }
}

impl From<u128> for InvokeArg {
    fn from(value: u128) -> Self {
        InvokeArg::Uint128(value)
    }
}

// Serialization

impl<'s> Serialize for InvokeContext<'s> {
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

        if let Some(gas) = self.gas {
            params_seq.serialize_element(&gas)?;
        }

        if let Some(system) = self.system {
            params_seq.serialize_element(&(system.as_ptr().as_ptr() as usize))?;
        }

        for arg in &self.args {
            params_seq.serialize_element(arg)?;
        }

        params_seq.end()
    }
}

impl Serialize for InvokeArg {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            InvokeArg::Felt252(value) => {
                let value = felt252_bigint(value.to_bigint());
                let mut seq = serializer.serialize_seq(Some(value.len()))?;
                for val in &value {
                    seq.serialize_element(val)?;
                }
                seq.end()
            }
            InvokeArg::Array(value) => {
                let mut seq = serializer.serialize_seq(Some(value.len()))?;
                for val in value {
                    seq.serialize_element(val)?;
                }
                seq.end()
            }
            InvokeArg::Struct(value) => {
                let mut seq = serializer.serialize_seq(Some(value.len()))?;
                for val in value {
                    seq.serialize_element(val)?;
                }
                seq.end()
            }
            InvokeArg::Enum { tag, value } => {
                let mut seq = serializer.serialize_seq(Some(2))?;
                seq.serialize_element(tag)?;
                seq.serialize_element(value.as_ref())?;
                seq.end()
            }
            InvokeArg::Box(value) => serializer.serialize_some(value.as_ref()),
            InvokeArg::Nullable(value) => match value {
                Some(value) => serializer.serialize_some(value.as_ref()),
                None => serializer.serialize_none(),
            },
            InvokeArg::Uint8(value) => serializer.serialize_u8(*value),
            InvokeArg::Uint16(value) => serializer.serialize_u16(*value),
            InvokeArg::Uint32(value) => serializer.serialize_u32(*value),
            InvokeArg::Uint64(value) => serializer.serialize_u64(*value),
            InvokeArg::Uint128(value) => serializer.serialize_u128(*value),
        }
    }
}
