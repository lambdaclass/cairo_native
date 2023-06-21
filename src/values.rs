use crate::{metadata::MetadataStorage, types::TypeBuilder};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{core::CoreTypeConcrete, GenericLibfunc, GenericType},
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{ir::Module, Context};
use num_bigint::BigUint;
use std::{alloc::Layout, cell::RefCell, error::Error, fmt, ptr::null_mut, slice};

pub mod array;
pub mod bitwise;
pub mod r#box;
pub mod builtin_costs;
pub mod ec_op;
pub mod ec_point;
pub mod ec_state;
pub mod r#enum;
pub mod felt252;
pub mod felt252_dict;
pub mod felt252_dict_entry;
pub mod gas_builtin;
pub mod non_zero;
pub mod nullable;
pub mod pedersen;
pub mod poseidon;
pub mod range_check;
pub mod segment_arena;
pub mod snapshot;
pub mod squashed_felt252_dict;
pub mod stark_net;
pub mod r#struct;
pub mod uint128;
pub mod uint128_mul_guarantee;
pub mod uint16;
pub mod uint32;
pub mod uint64;
pub mod uint8;
pub mod uninitialized;

pub trait ValueBuilder {
    type Error: Error;

    fn layout<TType, TLibfunc>(
        &self,
        context: &Context,
        module: &Module,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
    ) -> Layout
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder + ValueBuilder;

    fn alloc<TType, TLibfunc>(
        &self,
        arena: &Bump,
        context: &Context,
        module: &Module,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
    ) -> *mut ()
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder + ValueBuilder;

    unsafe fn parse(&self, target: *mut (), src: &str) -> Result<(), Self::Error>;

    unsafe fn debug<TType, TLibfunc>(
        &self,
        f: &mut fmt::Formatter,
        context: &Context,
        module: &Module,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
        source: *mut (),
    ) -> fmt::Result
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder + ValueBuilder;

    fn parsed<TType, TLibfunc>(
        &self,
        arena: &Bump,
        context: &Context,
        module: &Module,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
        src: &str,
    ) -> Result<*mut (), Self::Error>
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder + ValueBuilder,
    {
        let ptr = arena
            .alloc_layout(self.layout(context, module, registry, metadata))
            .as_ptr() as *mut ();
        unsafe {
            self.parse(ptr, src)?;
        }

        Ok(ptr)
    }
}

impl ValueBuilder for CoreTypeConcrete {
    type Error = std::convert::Infallible;

    fn layout<TType, TLibfunc>(
        &self,
        context: &Context,
        module: &Module,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
    ) -> Layout
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder + ValueBuilder,
    {
        let mlir_ty = self.build(context, module, registry, metadata).unwrap();
        Layout::from_size_align(
            crate::ffi::get_size(module, &mlir_ty),
            crate::ffi::get_abi_alignment(module, &mlir_ty),
        )
        .unwrap()
    }

    fn alloc<TType, TLibfunc>(
        &self,
        arena: &Bump,
        context: &Context,
        module: &Module,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
    ) -> *mut ()
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder + ValueBuilder,
    {
        match self {
            CoreTypeConcrete::Array(_) => todo!(),
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => todo!(),
            CoreTypeConcrete::GasBuiltin(_) => arena
                .alloc_layout(self.layout(context, module, registry, metadata))
                .as_ptr() as *mut (),
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => todo!(),
            CoreTypeConcrete::Uint16(_) => todo!(),
            CoreTypeConcrete::Uint32(_) => todo!(),
            CoreTypeConcrete::Uint64(_) => todo!(),
            CoreTypeConcrete::Uint128(_) => todo!(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => null_mut::<()>(),
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => arena
                .alloc_layout(self.layout(context, module, registry, metadata))
                .as_ptr() as *mut (),
            CoreTypeConcrete::Struct(_) => todo!(),
            CoreTypeConcrete::Felt252Dict(_) => todo!(),
            CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
            CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
            CoreTypeConcrete::Pedersen(_) => todo!(),
            CoreTypeConcrete::Poseidon(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(_) => todo!(),
            CoreTypeConcrete::SegmentArena(_) => todo!(),
            CoreTypeConcrete::Snapshot(_) => todo!(),
        }
    }

    unsafe fn parse(&self, target: *mut (), src: &str) -> Result<(), Self::Error> {
        match self {
            CoreTypeConcrete::Array(_) => todo!(),
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => todo!(),
            CoreTypeConcrete::GasBuiltin(_) => {
                (target as *mut usize).write(src.parse::<usize>().unwrap());
            }
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => todo!(),
            CoreTypeConcrete::Uint16(_) => todo!(),
            CoreTypeConcrete::Uint32(_) => todo!(),
            CoreTypeConcrete::Uint64(_) => todo!(),
            CoreTypeConcrete::Uint128(_) => todo!(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => todo!(),
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => todo!(),
            CoreTypeConcrete::Struct(_) => todo!(),
            CoreTypeConcrete::Felt252Dict(_) => todo!(),
            CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
            CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
            CoreTypeConcrete::Pedersen(_) => todo!(),
            CoreTypeConcrete::Poseidon(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(_) => todo!(),
            CoreTypeConcrete::SegmentArena(_) => todo!(),
            CoreTypeConcrete::Snapshot(_) => todo!(),
        }

        Ok(())
    }

    unsafe fn debug<TType, TLibfunc>(
        &self,
        f: &mut fmt::Formatter,
        context: &Context,
        module: &Module,
        registry: &ProgramRegistry<TType, TLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
        source: *mut (),
    ) -> fmt::Result
    where
        TType: GenericType<Concrete = Self>,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder + ValueBuilder,
    {
        match self {
            CoreTypeConcrete::Array(_) => todo!(),
            CoreTypeConcrete::Bitwise(_) => todo!(),
            CoreTypeConcrete::Box(_) => todo!(),
            CoreTypeConcrete::EcOp(_) => todo!(),
            CoreTypeConcrete::EcPoint(_) => todo!(),
            CoreTypeConcrete::EcState(_) => todo!(),
            CoreTypeConcrete::Felt252(_) => {
                write!(
                    f,
                    "{}",
                    BigUint::from_bytes_le(slice::from_raw_parts(source as *mut u8, 32))
                )
            }
            CoreTypeConcrete::GasBuiltin(_) => write!(f, "{}", (source as *mut usize).read()),
            CoreTypeConcrete::BuiltinCosts(_) => todo!(),
            CoreTypeConcrete::Uint8(_) => todo!(),
            CoreTypeConcrete::Uint16(_) => todo!(),
            CoreTypeConcrete::Uint32(_) => todo!(),
            CoreTypeConcrete::Uint64(_) => todo!(),
            CoreTypeConcrete::Uint128(_) => todo!(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
            CoreTypeConcrete::NonZero(_) => todo!(),
            CoreTypeConcrete::Nullable(_) => todo!(),
            CoreTypeConcrete::RangeCheck(_) => todo!(),
            CoreTypeConcrete::Uninitialized(_) => todo!(),
            CoreTypeConcrete::Enum(_) => {
                let payload_tys = self.variants().unwrap();
                let (tag_ty, _, align) = crate::types::r#enum::get_type_for_variants(
                    context,
                    module,
                    registry,
                    metadata,
                    payload_tys,
                )
                .unwrap();

                let tag_size = crate::ffi::get_size(module, &tag_ty);
                let layout = Layout::from_size_align(tag_size, align).unwrap();

                let index = match tag_size {
                    1 => (source as *mut u8).read() as usize,
                    2 => (source as *mut u16).read() as usize,
                    4 => (source as *mut u32).read() as usize,
                    8 => (source as *mut u64).read() as usize,
                    _ => panic!(),
                };

                let (_, offset) = layout
                    .extend(
                        registry
                            .get_type(&payload_tys[index])
                            .unwrap()
                            .layout(context, module, registry, metadata),
                    )
                    .unwrap();

                f.debug_tuple(&format!("{id}<{index}>"))
                    .field(&DebugWrapper {
                        inner: registry.get_type(&payload_tys[index]).unwrap(),
                        context,
                        module,
                        registry,
                        metadata: RefCell::new(metadata),
                        id: &payload_tys[index],
                        source: source.byte_offset(offset as isize),
                    })
                    .finish()
            }
            CoreTypeConcrete::Struct(info) => {
                let mut struct_fmt = f.debug_tuple(&format!("{id}"));

                let mut layout: Option<(Layout, usize)> = None;
                for member in info.members.iter() {
                    let member_layout = registry
                        .get_type(member)
                        .unwrap()
                        .layout(context, module, registry, metadata);

                    let (new_layout, offset) = match layout {
                        Some((layout, _)) => layout.extend(member_layout).unwrap(),
                        None => (member_layout, 0),
                    };

                    struct_fmt.field(&DebugWrapper {
                        inner: registry.get_type(member).unwrap(),
                        context,
                        module,
                        registry,
                        metadata: RefCell::new(metadata),
                        id: member,
                        source: source.byte_offset(offset as isize),
                    });

                    layout = Some((new_layout, offset));
                }

                struct_fmt.finish()
            }
            CoreTypeConcrete::Felt252Dict(_) => todo!(),
            CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
            CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
            CoreTypeConcrete::Pedersen(_) => todo!(),
            CoreTypeConcrete::Poseidon(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::StarkNet(_) => todo!(),
            CoreTypeConcrete::SegmentArena(_) => todo!(),
            CoreTypeConcrete::Snapshot(_) => todo!(),
        }
    }
}

pub struct DebugWrapper<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder + ValueBuilder,
{
    pub inner: &'a <TType as GenericType>::Concrete,
    pub context: &'a Context,
    pub module: &'a Module<'a>,
    pub registry: &'a ProgramRegistry<TType, TLibfunc>,
    pub metadata: RefCell<&'a mut MetadataStorage>,
    pub id: &'a ConcreteTypeId,
    pub source: *mut (),
}

impl<'a, TType, TLibfunc> fmt::Debug for DebugWrapper<'a, TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder + ValueBuilder,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            self.inner.debug(
                f,
                self.context,
                self.module,
                self.registry,
                *self.metadata.borrow_mut(),
                self.id,
                self.source,
            )
        }
    }
}
