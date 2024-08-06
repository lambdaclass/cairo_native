use crate::{
    error,
    starknet::{ArrayAbi, U256},
    types::TypeBuilder,
    values::JitValue,
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::{secp256::Secp256PointTypeConcrete, StarkNetTypeConcrete},
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use std::ptr::{null, NonNull};

mod aarch64;
mod x86_64;

/// Implemented by all supported argument types.
pub trait AbiArgument {
    /// Serialize the argument into the buffer. This method should keep track of arch-dependent
    /// stuff like register vs stack allocation.
    fn to_bytes(&self, buffer: &mut Vec<u8>) -> Result<(), error::Error>;
}

/// A wrapper that implements `AbiArgument` for `JitValue`s. It contains all the required stuff to
/// serialize all possible `JitValue`s.
pub struct JitValueWithInfoWrapper<'a> {
    pub value: &'a JitValue,
    pub type_id: &'a ConcreteTypeId,
    pub info: &'a CoreTypeConcrete,

    pub arena: &'a Bump,
    pub registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
}

impl<'a> JitValueWithInfoWrapper<'a> {
    fn map<'b>(
        &'b self,
        value: &'b JitValue,
        type_id: &'b ConcreteTypeId,
    ) -> Result<JitValueWithInfoWrapper<'b>, error::Error>
    where
        'b: 'a,
    {
        Ok(Self {
            value,
            type_id,
            info: self.registry.get_type(type_id)?,
            arena: self.arena,
            registry: self.registry,
        })
    }
}

impl<'a> AbiArgument for JitValueWithInfoWrapper<'a> {
    fn to_bytes(&self, buffer: &mut Vec<u8>) -> Result<(), error::Error> {
        match (self.value, self.info) {
            (value, CoreTypeConcrete::Box(info)) => {
                let ptr = value.to_jit(self.arena, self.registry, self.type_id)?;

                let layout = self.registry.get_type(&info.ty)?.layout(self.registry)?;
                let heap_ptr = unsafe {
                    let heap_ptr = libc::malloc(layout.size());
                    libc::memcpy(heap_ptr, ptr.as_ptr().cast(), layout.size());
                    heap_ptr
                };

                heap_ptr.to_bytes(buffer)?;
            }
            (value, CoreTypeConcrete::Nullable(info)) => {
                if matches!(value, JitValue::Null) {
                    null::<()>().to_bytes(buffer)?;
                } else {
                    let ptr = value.to_jit(self.arena, self.registry, self.type_id)?;

                    let layout = self.registry.get_type(&info.ty)?.layout(self.registry)?;
                    let heap_ptr = unsafe {
                        let heap_ptr = libc::malloc(layout.size());
                        libc::memcpy(heap_ptr, ptr.as_ptr().cast(), layout.size());
                        heap_ptr
                    };

                    heap_ptr.to_bytes(buffer)?;
                }
            }
            (value, CoreTypeConcrete::NonZero(info) | CoreTypeConcrete::Snapshot(info)) => {
                self.map(value, &info.ty)?.to_bytes(buffer)?
            }

            (JitValue::Array(_), CoreTypeConcrete::Array(_)) => {
                // TODO: Assert that `info.ty` matches all the values' types.

                let abi_ptr = self.value.to_jit(self.arena, self.registry, self.type_id)?;
                let abi = unsafe { abi_ptr.cast::<ArrayAbi<()>>().as_ref() };

                abi.ptr.to_bytes(buffer)?;
                abi.since.to_bytes(buffer)?;
                abi.until.to_bytes(buffer)?;
                abi.capacity.to_bytes(buffer)?;
            }
            (JitValue::BoundedInt { .. }, CoreTypeConcrete::BoundedInt(_)) => todo!(),
            (JitValue::Bytes31(value), CoreTypeConcrete::Bytes31(_)) => value.to_bytes(buffer)?,
            (JitValue::EcPoint(x, y), CoreTypeConcrete::EcPoint(_)) => {
                x.to_bytes(buffer)?;
                y.to_bytes(buffer)?;
            }
            (JitValue::EcState(x, y, x0, y0), CoreTypeConcrete::EcState(_)) => {
                x.to_bytes(buffer)?;
                y.to_bytes(buffer)?;
                x0.to_bytes(buffer)?;
                y0.to_bytes(buffer)?;
            }
            (JitValue::Enum { tag, value, .. }, CoreTypeConcrete::Enum(info)) => {
                if self.info.is_memory_allocated(self.registry) {
                    let abi_ptr = self.value.to_jit(self.arena, self.registry, self.type_id)?;

                    let abi_ptr = unsafe { *abi_ptr.cast::<NonNull<()>>().as_ref() };
                    abi_ptr.as_ptr().to_bytes(buffer)?;
                } else {
                    match (info.variants.len().next_power_of_two().trailing_zeros() + 7) / 8 {
                        0 => {}
                        _ => (*tag as u64).to_bytes(buffer)?,
                    }

                    self.map(value, &info.variants[*tag])?.to_bytes(buffer)?;
                }
            }
            (
                JitValue::Felt252(value),
                CoreTypeConcrete::Felt252(_)
                | CoreTypeConcrete::StarkNet(
                    StarkNetTypeConcrete::ClassHash(_)
                    | StarkNetTypeConcrete::ContractAddress(_)
                    | StarkNetTypeConcrete::StorageAddress(_)
                    | StarkNetTypeConcrete::StorageBaseAddress(_),
                ),
            ) => value.to_bytes(buffer)?,
            (JitValue::Felt252Dict { .. }, CoreTypeConcrete::Felt252Dict(_)) => {
                #[cfg(not(feature = "with-runtime"))]
                unimplemented!("enable the `with-runtime` feature to use felt252 dicts");

                // TODO: Assert that `info.ty` matches all the values' types.

                self.value
                    .to_jit(self.arena, self.registry, self.type_id)?
                    .as_ptr()
                    .to_bytes(buffer)?
            }
            (
                JitValue::Secp256K1Point { x, y },
                CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::Secp256Point(
                    Secp256PointTypeConcrete::K1(_),
                )),
            )
            | (
                JitValue::Secp256R1Point { x, y },
                CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::Secp256Point(
                    Secp256PointTypeConcrete::R1(_),
                )),
            ) => {
                let x = U256 { lo: x.0, hi: x.1 };
                let y = U256 { lo: y.0, hi: y.1 };

                x.to_bytes(buffer)?;
                y.to_bytes(buffer)?;
            }
            (JitValue::Sint128(value), CoreTypeConcrete::Sint128(_)) => value.to_bytes(buffer)?,
            (JitValue::Sint16(value), CoreTypeConcrete::Sint16(_)) => value.to_bytes(buffer)?,
            (JitValue::Sint32(value), CoreTypeConcrete::Sint32(_)) => value.to_bytes(buffer)?,
            (JitValue::Sint64(value), CoreTypeConcrete::Sint64(_)) => value.to_bytes(buffer)?,
            (JitValue::Sint8(value), CoreTypeConcrete::Sint8(_)) => value.to_bytes(buffer)?,
            (JitValue::Struct { fields, .. }, CoreTypeConcrete::Struct(info)) => {
                fields
                    .iter()
                    .zip(&info.members)
                    .map(|(value, type_id)| self.map(value, type_id))
                    .try_for_each(|wrapper| wrapper?.to_bytes(buffer))?;
            }
            (JitValue::Uint128(value), CoreTypeConcrete::Uint128(_)) => value.to_bytes(buffer)?,
            (JitValue::Uint16(value), CoreTypeConcrete::Uint16(_)) => value.to_bytes(buffer)?,
            (JitValue::Uint32(value), CoreTypeConcrete::Uint32(_)) => value.to_bytes(buffer)?,
            (JitValue::Uint64(value), CoreTypeConcrete::Uint64(_)) => value.to_bytes(buffer)?,
            (JitValue::Uint8(value), CoreTypeConcrete::Uint8(_)) => value.to_bytes(buffer)?,
            _ => todo!(
                "abi argument unimplemented for ({:?}, {:?})",
                self.value,
                self.type_id
            ),
        }

        Ok(())
    }
}
