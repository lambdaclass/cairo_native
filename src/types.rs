////! # Compiler type infrastructure
//! # Compiler type infrastructure
////!
//!
////! Contains type generation stuff (aka. conversion from Sierra to MLIR types).
//! Contains type generation stuff (aka. conversion from Sierra to MLIR types).
//

//use crate::{
use crate::{
//    error::Error as CoreTypeBuilderError,
    error::Error as CoreTypeBuilderError,
//    libfuncs::LibfuncHelper,
    libfuncs::LibfuncHelper,
//    metadata::{
    metadata::{
//        realloc_bindings::ReallocBindingsMeta, runtime_bindings::RuntimeBindingsMeta,
        realloc_bindings::ReallocBindingsMeta, runtime_bindings::RuntimeBindingsMeta,
//        MetadataStorage,
        MetadataStorage,
//    },
    },
//    utils::{get_integer_layout, layout_repeat, ProgramRegistryExt},
    utils::{get_integer_layout, layout_repeat, ProgramRegistryExt},
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
//        starknet::StarkNetTypeConcrete,
        starknet::StarkNetTypeConcrete,
//    },
    },
//    ids::{ConcreteTypeId, UserTypeId},
    ids::{ConcreteTypeId, UserTypeId},
//    program::GenericArg,
    program::GenericArg,
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{
    dialect::{
//        arith,
        arith,
//        llvm::{self, r#type::pointer},
        llvm::{self, r#type::pointer},
//        ods,
        ods,
//    },
    },
//    ir::{
    ir::{
//        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
//        r#type::IntegerType,
        r#type::IntegerType,
//        Block, Location, Module, Type, Value,
        Block, Location, Module, Type, Value,
//    },
    },
//    Context,
    Context,
//};
};
//use num_traits::Signed;
use num_traits::Signed;
//use std::{alloc::Layout, error::Error, ops::Deref, sync::OnceLock};
use std::{alloc::Layout, error::Error, ops::Deref, sync::OnceLock};
//

//pub mod array;
pub mod array;
//pub mod bitwise;
pub mod bitwise;
//pub mod bounded_int;
pub mod bounded_int;
//pub mod r#box;
pub mod r#box;
//pub mod builtin_costs;
pub mod builtin_costs;
//pub mod bytes31;
pub mod bytes31;
//pub mod ec_op;
pub mod ec_op;
//pub mod ec_point;
pub mod ec_point;
//pub mod ec_state;
pub mod ec_state;
//pub mod r#enum;
pub mod r#enum;
//pub mod felt252;
pub mod felt252;
//pub mod felt252_dict;
pub mod felt252_dict;
//pub mod felt252_dict_entry;
pub mod felt252_dict_entry;
//pub mod gas_builtin;
pub mod gas_builtin;
//pub mod non_zero;
pub mod non_zero;
//pub mod nullable;
pub mod nullable;
//pub mod pedersen;
pub mod pedersen;
//pub mod poseidon;
pub mod poseidon;
//pub mod range_check;
pub mod range_check;
//pub mod segment_arena;
pub mod segment_arena;
//pub mod snapshot;
pub mod snapshot;
//pub mod squashed_felt252_dict;
pub mod squashed_felt252_dict;
//pub mod starknet;
pub mod starknet;
//pub mod r#struct;
pub mod r#struct;
//pub mod uint128;
pub mod uint128;
//pub mod uint128_mul_guarantee;
pub mod uint128_mul_guarantee;
//pub mod uint16;
pub mod uint16;
//pub mod uint32;
pub mod uint32;
//pub mod uint64;
pub mod uint64;
//pub mod uint8;
pub mod uint8;
//pub mod uninitialized;
pub mod uninitialized;
//

///// Generation of MLIR types from their Sierra counterparts.
/// Generation of MLIR types from their Sierra counterparts.
/////
///
///// All possible Sierra types must implement it. It is already implemented for all the core Sierra
/// All possible Sierra types must implement it. It is already implemented for all the core Sierra
///// types, contained in [CoreTypeConcrete].
/// types, contained in [CoreTypeConcrete].
//pub trait TypeBuilder {
pub trait TypeBuilder {
//    /// Error type returned by this trait's methods.
    /// Error type returned by this trait's methods.
//    type Error: Error;
    type Error: Error;
//

//    /// Build the MLIR type.
    /// Build the MLIR type.
//    fn build<'ctx>(
    fn build<'ctx>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        module: &Module<'ctx>,
        module: &Module<'ctx>,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        metadata: &mut MetadataStorage,
        metadata: &mut MetadataStorage,
//        self_ty: &ConcreteTypeId,
        self_ty: &ConcreteTypeId,
//    ) -> Result<Type<'ctx>, Self::Error>;
    ) -> Result<Type<'ctx>, Self::Error>;
//

//    /// Return whether the type is a builtin.
    /// Return whether the type is a builtin.
//    fn is_builtin(&self) -> bool;
    fn is_builtin(&self) -> bool;
//    /// Return whether the type requires a return pointer when returning.
    /// Return whether the type requires a return pointer when returning.
//    fn is_complex(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool;
    fn is_complex(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool;
//    /// Return whether the Sierra type resolves to a zero-sized type.
    /// Return whether the Sierra type resolves to a zero-sized type.
//    fn is_zst(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool;
    fn is_zst(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool;
//

//    /// Generate the layout of the MLIR type.
    /// Generate the layout of the MLIR type.
//    ///
    ///
//    /// Used in both the compiler and the interface when calling the compiled code.
    /// Used in both the compiler and the interface when calling the compiled code.
//    fn layout(
    fn layout(
//        &self,
        &self,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    ) -> Result<Layout, Self::Error>;
    ) -> Result<Layout, Self::Error>;
//

//    /// Whether the layout should be allocated in memory (either the stack or the heap) when used as
    /// Whether the layout should be allocated in memory (either the stack or the heap) when used as
//    /// a function invocation argument or return value.
    /// a function invocation argument or return value.
//    fn is_memory_allocated(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool;
    fn is_memory_allocated(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool;
//

//    /// If the type is an integer type, return its width in bits.
    /// If the type is an integer type, return its width in bits.
//    ///
    ///
//    /// TODO: How is it used?
    /// TODO: How is it used?
//    fn integer_width(&self) -> Option<usize>;
    fn integer_width(&self) -> Option<usize>;
//

//    /// If the type is an integer type, return if its signed.
    /// If the type is an integer type, return if its signed.
//    fn is_integer_signed(&self) -> Option<bool>;
    fn is_integer_signed(&self) -> Option<bool>;
//

//    /// If the type is a enum type, return all possible variants.
    /// If the type is a enum type, return all possible variants.
//    ///
    ///
//    /// TODO: How is it used?
    /// TODO: How is it used?
//    fn variants(&self) -> Option<&[ConcreteTypeId]>;
    fn variants(&self) -> Option<&[ConcreteTypeId]>;
//

//    // If the type is a struct, return the field types.
    // If the type is a struct, return the field types.
//    fn fields(&self) -> Option<&[ConcreteTypeId]>;
    fn fields(&self) -> Option<&[ConcreteTypeId]>;
//

//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    fn build_default<'ctx, 'this>(
    fn build_default<'ctx, 'this>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        entry: &'this Block<'ctx>,
        entry: &'this Block<'ctx>,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        helper: &LibfuncHelper<'ctx, 'this>,
        helper: &LibfuncHelper<'ctx, 'this>,
//        metadata: &mut MetadataStorage,
        metadata: &mut MetadataStorage,
//        self_ty: &ConcreteTypeId,
        self_ty: &ConcreteTypeId,
//    ) -> Result<Value<'ctx, 'this>, Self::Error>;
    ) -> Result<Value<'ctx, 'this>, Self::Error>;
//

//    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
//    fn build_drop<'ctx, 'this>(
    fn build_drop<'ctx, 'this>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        entry: &'this Block<'ctx>,
        entry: &'this Block<'ctx>,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        helper: &LibfuncHelper<'ctx, 'this>,
        helper: &LibfuncHelper<'ctx, 'this>,
//        metadata: &mut MetadataStorage,
        metadata: &mut MetadataStorage,
//        self_ty: &ConcreteTypeId,
        self_ty: &ConcreteTypeId,
//    ) -> Result<(), Self::Error>;
    ) -> Result<(), Self::Error>;
//}
}
//

//impl TypeBuilder for CoreTypeConcrete {
impl TypeBuilder for CoreTypeConcrete {
//    type Error = CoreTypeBuilderError;
    type Error = CoreTypeBuilderError;
//

//    fn build<'ctx>(
    fn build<'ctx>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        module: &Module<'ctx>,
        module: &Module<'ctx>,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        metadata: &mut MetadataStorage,
        metadata: &mut MetadataStorage,
//        self_ty: &ConcreteTypeId,
        self_ty: &ConcreteTypeId,
//    ) -> Result<Type<'ctx>, Self::Error> {
    ) -> Result<Type<'ctx>, Self::Error> {
//        match self {
        match self {
//            Self::Array(info) => self::array::build(
            Self::Array(info) => self::array::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Bitwise(info) => self::bitwise::build(
            Self::Bitwise(info) => self::bitwise::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::BoundedInt(info) => self::bounded_int::build(
            Self::BoundedInt(info) => self::bounded_int::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Box(info) => self::r#box::build(
            Self::Box(info) => self::r#box::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Bytes31(info) => self::bytes31::build(context, module, registry, metadata, info),
            Self::Bytes31(info) => self::bytes31::build(context, module, registry, metadata, info),
//            Self::BuiltinCosts(info) => self::builtin_costs::build(
            Self::BuiltinCosts(info) => self::builtin_costs::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Const(_) => todo!(),
            Self::Const(_) => todo!(),
//            Self::EcOp(info) => self::ec_op::build(
            Self::EcOp(info) => self::ec_op::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::EcPoint(info) => self::ec_point::build(
            Self::EcPoint(info) => self::ec_point::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::EcState(info) => self::ec_state::build(
            Self::EcState(info) => self::ec_state::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Enum(info) => self::r#enum::build(
            Self::Enum(info) => self::r#enum::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Felt252(info) => self::felt252::build(
            Self::Felt252(info) => self::felt252::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Felt252Dict(info) => self::felt252_dict::build(
            Self::Felt252Dict(info) => self::felt252_dict::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Felt252DictEntry(info) => self::felt252_dict_entry::build(
            Self::Felt252DictEntry(info) => self::felt252_dict_entry::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::GasBuiltin(info) => self::gas_builtin::build(
            Self::GasBuiltin(info) => self::gas_builtin::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::NonZero(info) => self::non_zero::build(
            Self::NonZero(info) => self::non_zero::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Nullable(info) => self::nullable::build(
            Self::Nullable(info) => self::nullable::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Pedersen(info) => self::pedersen::build(
            Self::Pedersen(info) => self::pedersen::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Poseidon(info) => self::poseidon::build(
            Self::Poseidon(info) => self::poseidon::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::RangeCheck(info) => self::range_check::build(
            Self::RangeCheck(info) => self::range_check::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::SegmentArena(info) => self::segment_arena::build(
            Self::SegmentArena(info) => self::segment_arena::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Sint8(info) => self::uint8::build(
            Self::Sint8(info) => self::uint8::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Sint16(info) => self::uint16::build(
            Self::Sint16(info) => self::uint16::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Sint32(info) => self::uint32::build(
            Self::Sint32(info) => self::uint32::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Sint64(info) => self::uint64::build(
            Self::Sint64(info) => self::uint64::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Sint128(info) => self::uint128::build(
            Self::Sint128(info) => self::uint128::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Snapshot(info) => self::snapshot::build(
            Self::Snapshot(info) => self::snapshot::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Span(_) => todo!("implement span type"),
            Self::Span(_) => todo!("implement span type"),
//            Self::SquashedFelt252Dict(info) => self::squashed_felt252_dict::build(
            Self::SquashedFelt252Dict(info) => self::squashed_felt252_dict::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::StarkNet(selector) => self::starknet::build(
            Self::StarkNet(selector) => self::starknet::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, selector),
                WithSelf::new(self_ty, selector),
//            ),
            ),
//            Self::Struct(info) => self::r#struct::build(
            Self::Struct(info) => self::r#struct::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Uint128(info) => self::uint128::build(
            Self::Uint128(info) => self::uint128::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Uint128MulGuarantee(info) => self::uint128_mul_guarantee::build(
            Self::Uint128MulGuarantee(info) => self::uint128_mul_guarantee::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Uint16(info) => self::uint16::build(
            Self::Uint16(info) => self::uint16::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Uint32(info) => self::uint32::build(
            Self::Uint32(info) => self::uint32::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Uint64(info) => self::uint64::build(
            Self::Uint64(info) => self::uint64::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Uint8(info) => self::uint8::build(
            Self::Uint8(info) => self::uint8::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            Self::Uninitialized(info) => self::uninitialized::build(
            Self::Uninitialized(info) => self::uninitialized::build(
//                context,
                context,
//                module,
                module,
//                registry,
                registry,
//                metadata,
                metadata,
//                WithSelf::new(self_ty, info),
                WithSelf::new(self_ty, info),
//            ),
            ),
//            CoreTypeConcrete::Coupon(_) => todo!(),
            CoreTypeConcrete::Coupon(_) => todo!(),
//        }
        }
//    }
    }
//

//    fn is_builtin(&self) -> bool {
    fn is_builtin(&self) -> bool {
//        matches!(
        matches!(
//            self,
            self,
//            CoreTypeConcrete::Bitwise(_)
            CoreTypeConcrete::Bitwise(_)
//                | CoreTypeConcrete::EcOp(_)
                | CoreTypeConcrete::EcOp(_)
//                | CoreTypeConcrete::GasBuiltin(_)
                | CoreTypeConcrete::GasBuiltin(_)
//                | CoreTypeConcrete::BuiltinCosts(_)
                | CoreTypeConcrete::BuiltinCosts(_)
//                | CoreTypeConcrete::RangeCheck(_)
                | CoreTypeConcrete::RangeCheck(_)
//                | CoreTypeConcrete::Pedersen(_)
                | CoreTypeConcrete::Pedersen(_)
//                | CoreTypeConcrete::Poseidon(_)
                | CoreTypeConcrete::Poseidon(_)
//                | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_))
                | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_))
//                | CoreTypeConcrete::SegmentArena(_)
                | CoreTypeConcrete::SegmentArena(_)
//        )
        )
//    }
    }
//

//    fn is_complex(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool {
    fn is_complex(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool {
//        match self {
        match self {
//            // Builtins.
            // Builtins.
//            CoreTypeConcrete::Bitwise(_)
            CoreTypeConcrete::Bitwise(_)
//            | CoreTypeConcrete::EcOp(_)
            | CoreTypeConcrete::EcOp(_)
//            | CoreTypeConcrete::GasBuiltin(_) // u128 is not complex
            | CoreTypeConcrete::GasBuiltin(_) // u128 is not complex
//            | CoreTypeConcrete::BuiltinCosts(_)
            | CoreTypeConcrete::BuiltinCosts(_)
//            | CoreTypeConcrete::RangeCheck(_)
            | CoreTypeConcrete::RangeCheck(_)
//            | CoreTypeConcrete::Pedersen(_)
            | CoreTypeConcrete::Pedersen(_)
//            | CoreTypeConcrete::Poseidon(_)
            | CoreTypeConcrete::Poseidon(_)
//            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) // u64 is not complex
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_)) // u64 is not complex
//            | CoreTypeConcrete::SegmentArena(_) => false,
            | CoreTypeConcrete::SegmentArena(_) => false,
//

//            CoreTypeConcrete::Box(_)
            CoreTypeConcrete::Box(_)
//            | CoreTypeConcrete::Uint8(_)
            | CoreTypeConcrete::Uint8(_)
//            | CoreTypeConcrete::Uint16(_)
            | CoreTypeConcrete::Uint16(_)
//            | CoreTypeConcrete::Uint32(_)
            | CoreTypeConcrete::Uint32(_)
//            | CoreTypeConcrete::Uint64(_)
            | CoreTypeConcrete::Uint64(_)
//            | CoreTypeConcrete::Uint128(_)
            | CoreTypeConcrete::Uint128(_)
//            | CoreTypeConcrete::Uint128MulGuarantee(_)
            | CoreTypeConcrete::Uint128MulGuarantee(_)
//            | CoreTypeConcrete::Sint8(_)
            | CoreTypeConcrete::Sint8(_)
//            | CoreTypeConcrete::Sint16(_)
            | CoreTypeConcrete::Sint16(_)
//            | CoreTypeConcrete::Sint32(_)
            | CoreTypeConcrete::Sint32(_)
//            | CoreTypeConcrete::Sint64(_)
            | CoreTypeConcrete::Sint64(_)
//            | CoreTypeConcrete::Sint128(_)
            | CoreTypeConcrete::Sint128(_)
//            | CoreTypeConcrete::Nullable(_)
            | CoreTypeConcrete::Nullable(_)
//            | CoreTypeConcrete::Felt252Dict(_)
            | CoreTypeConcrete::Felt252Dict(_)
//            | CoreTypeConcrete::SquashedFelt252Dict(_) => false,
            | CoreTypeConcrete::SquashedFelt252Dict(_) => false,
//

//            CoreTypeConcrete::Array(_) => true,
            CoreTypeConcrete::Array(_) => true,
//            CoreTypeConcrete::EcPoint(_) => true,
            CoreTypeConcrete::EcPoint(_) => true,
//            CoreTypeConcrete::EcState(_) => true,
            CoreTypeConcrete::EcState(_) => true,
//            CoreTypeConcrete::Felt252DictEntry(_) => true,
            CoreTypeConcrete::Felt252DictEntry(_) => true,
//

//            CoreTypeConcrete::Felt252(_)
            CoreTypeConcrete::Felt252(_)
//            | CoreTypeConcrete::Bytes31(_)
            | CoreTypeConcrete::Bytes31(_)
//            | CoreTypeConcrete::StarkNet(
            | CoreTypeConcrete::StarkNet(
//                StarkNetTypeConcrete::ClassHash(_)
                StarkNetTypeConcrete::ClassHash(_)
//                | StarkNetTypeConcrete::ContractAddress(_)
                | StarkNetTypeConcrete::ContractAddress(_)
//                | StarkNetTypeConcrete::StorageAddress(_)
                | StarkNetTypeConcrete::StorageAddress(_)
//                | StarkNetTypeConcrete::StorageBaseAddress(_)
                | StarkNetTypeConcrete::StorageBaseAddress(_)
//            ) => {
            ) => {
//                #[cfg(target_arch = "x86_64")]
                #[cfg(target_arch = "x86_64")]
//                let value = true;
                let value = true;
//

//                #[cfg(target_arch = "aarch64")]
                #[cfg(target_arch = "aarch64")]
//                let value = false;
                let value = false;
//

//                value
                value
//            },
            },
//

//            CoreTypeConcrete::NonZero(info)
            CoreTypeConcrete::NonZero(info)
//            | CoreTypeConcrete::Uninitialized(info)
            | CoreTypeConcrete::Uninitialized(info)
//            | CoreTypeConcrete::Snapshot(info) => registry.get_type(&info.ty).unwrap().is_complex(registry),
            | CoreTypeConcrete::Snapshot(info) => registry.get_type(&info.ty).unwrap().is_complex(registry),
//

//            CoreTypeConcrete::Enum(info) => match info.variants.len() {
            CoreTypeConcrete::Enum(info) => match info.variants.len() {
//                0 => false,
                0 => false,
//                1 => registry.get_type(&info.variants[0]).unwrap().is_complex(registry),
                1 => registry.get_type(&info.variants[0]).unwrap().is_complex(registry),
//                _ => !self.is_zst(registry),
                _ => !self.is_zst(registry),
//            },
            },
//            CoreTypeConcrete::Struct(_) => true,
            CoreTypeConcrete::Struct(_) => true,
//

//            CoreTypeConcrete::BoundedInt(_) => todo!(),
            CoreTypeConcrete::BoundedInt(_) => todo!(),
//            CoreTypeConcrete::Const(_) => todo!(),
            CoreTypeConcrete::Const(_) => todo!(),
//            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
//            CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::Secp256Point(_)) => todo!(),
            CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::Secp256Point(_)) => todo!(),
//            CoreTypeConcrete::Coupon(_) => todo!(),
            CoreTypeConcrete::Coupon(_) => todo!(),
//        }
        }
//    }
    }
//

//    fn is_zst(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool {
    fn is_zst(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool {
//        match self {
        match self {
//            // Builtin counters:
            // Builtin counters:
//            CoreTypeConcrete::Bitwise(_)
            CoreTypeConcrete::Bitwise(_)
//            | CoreTypeConcrete::EcOp(_)
            | CoreTypeConcrete::EcOp(_)
//            | CoreTypeConcrete::RangeCheck(_)
            | CoreTypeConcrete::RangeCheck(_)
//            | CoreTypeConcrete::Pedersen(_)
            | CoreTypeConcrete::Pedersen(_)
//            | CoreTypeConcrete::Poseidon(_)
            | CoreTypeConcrete::Poseidon(_)
//            | CoreTypeConcrete::SegmentArena(_) => false,
            | CoreTypeConcrete::SegmentArena(_) => false,
//            // Other builtins:
            // Other builtins:
//            CoreTypeConcrete::BuiltinCosts(_) | CoreTypeConcrete::Uint128MulGuarantee(_) => true,
            CoreTypeConcrete::BuiltinCosts(_) | CoreTypeConcrete::Uint128MulGuarantee(_) => true,
//

//            // Normal types:
            // Normal types:
//            CoreTypeConcrete::Array(_)
            CoreTypeConcrete::Array(_)
//            | CoreTypeConcrete::Box(_)
            | CoreTypeConcrete::Box(_)
//            | CoreTypeConcrete::Bytes31(_)
            | CoreTypeConcrete::Bytes31(_)
//            | CoreTypeConcrete::EcPoint(_)
            | CoreTypeConcrete::EcPoint(_)
//            | CoreTypeConcrete::EcState(_)
            | CoreTypeConcrete::EcState(_)
//            | CoreTypeConcrete::Felt252(_)
            | CoreTypeConcrete::Felt252(_)
//            | CoreTypeConcrete::GasBuiltin(_)
            | CoreTypeConcrete::GasBuiltin(_)
//            | CoreTypeConcrete::Uint8(_)
            | CoreTypeConcrete::Uint8(_)
//            | CoreTypeConcrete::Uint16(_)
            | CoreTypeConcrete::Uint16(_)
//            | CoreTypeConcrete::Uint32(_)
            | CoreTypeConcrete::Uint32(_)
//            | CoreTypeConcrete::Uint64(_)
            | CoreTypeConcrete::Uint64(_)
//            | CoreTypeConcrete::Uint128(_)
            | CoreTypeConcrete::Uint128(_)
//            | CoreTypeConcrete::Sint8(_)
            | CoreTypeConcrete::Sint8(_)
//            | CoreTypeConcrete::Sint16(_)
            | CoreTypeConcrete::Sint16(_)
//            | CoreTypeConcrete::Sint32(_)
            | CoreTypeConcrete::Sint32(_)
//            | CoreTypeConcrete::Sint64(_)
            | CoreTypeConcrete::Sint64(_)
//            | CoreTypeConcrete::Sint128(_)
            | CoreTypeConcrete::Sint128(_)
//            | CoreTypeConcrete::Felt252Dict(_)
            | CoreTypeConcrete::Felt252Dict(_)
//            | CoreTypeConcrete::Felt252DictEntry(_)
            | CoreTypeConcrete::Felt252DictEntry(_)
//            | CoreTypeConcrete::SquashedFelt252Dict(_)
            | CoreTypeConcrete::SquashedFelt252Dict(_)
//            | CoreTypeConcrete::StarkNet(_)
            | CoreTypeConcrete::StarkNet(_)
//            | CoreTypeConcrete::Nullable(_) => false,
            | CoreTypeConcrete::Nullable(_) => false,
//

//            // Containers:
            // Containers:
//            CoreTypeConcrete::NonZero(info)
            CoreTypeConcrete::NonZero(info)
//            | CoreTypeConcrete::Uninitialized(info)
            | CoreTypeConcrete::Uninitialized(info)
//            | CoreTypeConcrete::Snapshot(info) => {
            | CoreTypeConcrete::Snapshot(info) => {
//                let type_info = registry.get_type(&info.ty).unwrap();
                let type_info = registry.get_type(&info.ty).unwrap();
//                type_info.is_zst(registry)
                type_info.is_zst(registry)
//            }
            }
//

//            // Enums and structs:
            // Enums and structs:
//            CoreTypeConcrete::Enum(info) => {
            CoreTypeConcrete::Enum(info) => {
//                info.variants.is_empty()
                info.variants.is_empty()
//                    || (info.variants.len() == 1
                    || (info.variants.len() == 1
//                        && registry
                        && registry
//                            .get_type(&info.variants[0])
                            .get_type(&info.variants[0])
//                            .unwrap()
                            .unwrap()
//                            .is_zst(registry))
                            .is_zst(registry))
//            }
            }
//            CoreTypeConcrete::Struct(info) => info
            CoreTypeConcrete::Struct(info) => info
//                .members
                .members
//                .iter()
                .iter()
//                .all(|id| registry.get_type(id).unwrap().is_zst(registry)),
                .all(|id| registry.get_type(id).unwrap().is_zst(registry)),
//

//            CoreTypeConcrete::BoundedInt(_) => false,
            CoreTypeConcrete::BoundedInt(_) => false,
//            CoreTypeConcrete::Const(info) => {
            CoreTypeConcrete::Const(info) => {
//                let type_info = registry.get_type(&info.inner_ty).unwrap();
                let type_info = registry.get_type(&info.inner_ty).unwrap();
//                type_info.is_zst(registry)
                type_info.is_zst(registry)
//            }
            }
//            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
//            CoreTypeConcrete::Coupon(_) => todo!(),
            CoreTypeConcrete::Coupon(_) => todo!(),
//        }
        }
//    }
    }
//

//    fn layout(
    fn layout(
//        &self,
        &self,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    ) -> Result<Layout, Self::Error> {
    ) -> Result<Layout, Self::Error> {
//        Ok(match self {
        Ok(match self {
//            CoreTypeConcrete::Array(_) => {
            CoreTypeConcrete::Array(_) => {
//                Layout::new::<*mut ()>()
                Layout::new::<*mut ()>()
//                    .extend(get_integer_layout(32))?
                    .extend(get_integer_layout(32))?
//                    .0
                    .0
//                    .extend(get_integer_layout(32))?
                    .extend(get_integer_layout(32))?
//                    .0
                    .0
//                    .extend(get_integer_layout(32))?
                    .extend(get_integer_layout(32))?
//                    .0
                    .0
//            }
            }
//            CoreTypeConcrete::Bitwise(_) => Layout::new::<u64>(),
            CoreTypeConcrete::Bitwise(_) => Layout::new::<u64>(),
//            CoreTypeConcrete::Box(_) => Layout::new::<*mut ()>(),
            CoreTypeConcrete::Box(_) => Layout::new::<*mut ()>(),
//            CoreTypeConcrete::EcOp(_) => Layout::new::<u64>(),
            CoreTypeConcrete::EcOp(_) => Layout::new::<u64>(),
//            CoreTypeConcrete::EcPoint(_) => layout_repeat(&get_integer_layout(252), 2)?.0,
            CoreTypeConcrete::EcPoint(_) => layout_repeat(&get_integer_layout(252), 2)?.0,
//            CoreTypeConcrete::EcState(_) => layout_repeat(&get_integer_layout(252), 4)?.0,
            CoreTypeConcrete::EcState(_) => layout_repeat(&get_integer_layout(252), 4)?.0,
//            CoreTypeConcrete::Felt252(_) => get_integer_layout(252),
            CoreTypeConcrete::Felt252(_) => get_integer_layout(252),
//            CoreTypeConcrete::GasBuiltin(_) => get_integer_layout(128),
            CoreTypeConcrete::GasBuiltin(_) => get_integer_layout(128),
//            CoreTypeConcrete::BuiltinCosts(_) => Layout::new::<()>(),
            CoreTypeConcrete::BuiltinCosts(_) => Layout::new::<()>(),
//            CoreTypeConcrete::Uint8(_) => get_integer_layout(8),
            CoreTypeConcrete::Uint8(_) => get_integer_layout(8),
//            CoreTypeConcrete::Uint16(_) => get_integer_layout(16),
            CoreTypeConcrete::Uint16(_) => get_integer_layout(16),
//            CoreTypeConcrete::Uint32(_) => get_integer_layout(32),
            CoreTypeConcrete::Uint32(_) => get_integer_layout(32),
//            CoreTypeConcrete::Uint64(_) => get_integer_layout(64),
            CoreTypeConcrete::Uint64(_) => get_integer_layout(64),
//            CoreTypeConcrete::Uint128(_) => get_integer_layout(128),
            CoreTypeConcrete::Uint128(_) => get_integer_layout(128),
//            CoreTypeConcrete::Uint128MulGuarantee(_) => Layout::new::<()>(),
            CoreTypeConcrete::Uint128MulGuarantee(_) => Layout::new::<()>(),
//            CoreTypeConcrete::NonZero(info) => registry.get_type(&info.ty)?.layout(registry)?,
            CoreTypeConcrete::NonZero(info) => registry.get_type(&info.ty)?.layout(registry)?,
//            CoreTypeConcrete::Nullable(_) => Layout::new::<*mut ()>(),
            CoreTypeConcrete::Nullable(_) => Layout::new::<*mut ()>(),
//            CoreTypeConcrete::RangeCheck(_) => Layout::new::<u64>(),
            CoreTypeConcrete::RangeCheck(_) => Layout::new::<u64>(),
//            CoreTypeConcrete::Uninitialized(info) => {
            CoreTypeConcrete::Uninitialized(info) => {
//                registry.get_type(&info.ty)?.layout(registry)?
                registry.get_type(&info.ty)?.layout(registry)?
//            }
            }
//            CoreTypeConcrete::Enum(info) => {
            CoreTypeConcrete::Enum(info) => {
//                let tag_layout =
                let tag_layout =
//                    get_integer_layout(info.variants.len().next_power_of_two().trailing_zeros());
                    get_integer_layout(info.variants.len().next_power_of_two().trailing_zeros());
//

//                info.variants.iter().try_fold(tag_layout, |acc, id| {
                info.variants.iter().try_fold(tag_layout, |acc, id| {
//                    let layout = tag_layout
                    let layout = tag_layout
//                        .extend(registry.get_type(id)?.layout(registry)?)?
                        .extend(registry.get_type(id)?.layout(registry)?)?
//                        .0;
                        .0;
//

//                    Result::<_, Self::Error>::Ok(Layout::from_size_align(
                    Result::<_, Self::Error>::Ok(Layout::from_size_align(
//                        acc.size().max(layout.size()),
                        acc.size().max(layout.size()),
//                        acc.align().max(layout.align()),
                        acc.align().max(layout.align()),
//                    )?)
                    )?)
//                })?
                })?
//            }
            }
//            CoreTypeConcrete::Struct(info) => info
            CoreTypeConcrete::Struct(info) => info
//                .members
                .members
//                .iter()
                .iter()
//                .try_fold(Option::<Layout>::None, |acc, id| {
                .try_fold(Option::<Layout>::None, |acc, id| {
//                    Result::<_, Self::Error>::Ok(Some(match acc {
                    Result::<_, Self::Error>::Ok(Some(match acc {
//                        Some(layout) => layout.extend(registry.get_type(id)?.layout(registry)?)?.0,
                        Some(layout) => layout.extend(registry.get_type(id)?.layout(registry)?)?.0,
//                        None => registry.get_type(id)?.layout(registry)?,
                        None => registry.get_type(id)?.layout(registry)?,
//                    }))
                    }))
//                })?
                })?
//                .unwrap_or(Layout::from_size_align(0, 1)?),
                .unwrap_or(Layout::from_size_align(0, 1)?),
//            CoreTypeConcrete::Felt252Dict(_) => Layout::new::<*mut std::ffi::c_void>(), // ptr
            CoreTypeConcrete::Felt252Dict(_) => Layout::new::<*mut std::ffi::c_void>(), // ptr
//            CoreTypeConcrete::Felt252DictEntry(_) => {
            CoreTypeConcrete::Felt252DictEntry(_) => {
//                get_integer_layout(252)
                get_integer_layout(252)
//                    .extend(Layout::new::<*mut std::ffi::c_void>())
                    .extend(Layout::new::<*mut std::ffi::c_void>())
//                    .unwrap()
                    .unwrap()
//                    .0
                    .0
//                    .extend(Layout::new::<*mut std::ffi::c_void>())
                    .extend(Layout::new::<*mut std::ffi::c_void>())
//                    .unwrap()
                    .unwrap()
//                    .0
                    .0
//            }
            }
//            CoreTypeConcrete::SquashedFelt252Dict(_) => Layout::new::<*mut std::ffi::c_void>(), // ptr
            CoreTypeConcrete::SquashedFelt252Dict(_) => Layout::new::<*mut std::ffi::c_void>(), // ptr
//            CoreTypeConcrete::Pedersen(_) => Layout::new::<u64>(),
            CoreTypeConcrete::Pedersen(_) => Layout::new::<u64>(),
//            CoreTypeConcrete::Poseidon(_) => Layout::new::<u64>(),
            CoreTypeConcrete::Poseidon(_) => Layout::new::<u64>(),
//            CoreTypeConcrete::Span(_) => todo!(),
            CoreTypeConcrete::Span(_) => todo!(),
//            CoreTypeConcrete::StarkNet(info) => match info {
            CoreTypeConcrete::StarkNet(info) => match info {
//                StarkNetTypeConcrete::ClassHash(_) => get_integer_layout(252),
                StarkNetTypeConcrete::ClassHash(_) => get_integer_layout(252),
//                StarkNetTypeConcrete::ContractAddress(_) => get_integer_layout(252),
                StarkNetTypeConcrete::ContractAddress(_) => get_integer_layout(252),
//                StarkNetTypeConcrete::StorageBaseAddress(_) => get_integer_layout(252),
                StarkNetTypeConcrete::StorageBaseAddress(_) => get_integer_layout(252),
//                StarkNetTypeConcrete::StorageAddress(_) => get_integer_layout(252),
                StarkNetTypeConcrete::StorageAddress(_) => get_integer_layout(252),
//                StarkNetTypeConcrete::System(_) => Layout::new::<*mut ()>(),
                StarkNetTypeConcrete::System(_) => Layout::new::<*mut ()>(),
//                StarkNetTypeConcrete::Secp256Point(_) => {
                StarkNetTypeConcrete::Secp256Point(_) => {
//                    get_integer_layout(256)
                    get_integer_layout(256)
//                        .extend(get_integer_layout(256))
                        .extend(get_integer_layout(256))
//                        .unwrap()
                        .unwrap()
//                        .0
                        .0
//                }
                }
//            },
            },
//            CoreTypeConcrete::SegmentArena(_) => Layout::new::<u64>(),
            CoreTypeConcrete::SegmentArena(_) => Layout::new::<u64>(),
//            CoreTypeConcrete::Snapshot(info) => registry.get_type(&info.ty)?.layout(registry)?,
            CoreTypeConcrete::Snapshot(info) => registry.get_type(&info.ty)?.layout(registry)?,
//            CoreTypeConcrete::Sint8(_) => get_integer_layout(8),
            CoreTypeConcrete::Sint8(_) => get_integer_layout(8),
//            CoreTypeConcrete::Sint16(_) => get_integer_layout(16),
            CoreTypeConcrete::Sint16(_) => get_integer_layout(16),
//            CoreTypeConcrete::Sint32(_) => get_integer_layout(32),
            CoreTypeConcrete::Sint32(_) => get_integer_layout(32),
//            CoreTypeConcrete::Sint64(_) => get_integer_layout(64),
            CoreTypeConcrete::Sint64(_) => get_integer_layout(64),
//            CoreTypeConcrete::Sint128(_) => get_integer_layout(128),
            CoreTypeConcrete::Sint128(_) => get_integer_layout(128),
//            CoreTypeConcrete::Bytes31(_) => get_integer_layout(248),
            CoreTypeConcrete::Bytes31(_) => get_integer_layout(248),
//            CoreTypeConcrete::BoundedInt(info) => get_integer_layout(
            CoreTypeConcrete::BoundedInt(info) => get_integer_layout(
//                (info.range.lower.bits().max(info.range.upper.bits()) + 1)
                (info.range.lower.bits().max(info.range.upper.bits()) + 1)
//                    .try_into()
                    .try_into()
//                    .expect("should always fit u32"),
                    .expect("should always fit u32"),
//            ),
            ),
//            CoreTypeConcrete::Const(const_type) => {
            CoreTypeConcrete::Const(const_type) => {
//                registry.get_type(&const_type.inner_ty)?.layout(registry)?
                registry.get_type(&const_type.inner_ty)?.layout(registry)?
//            }
            }
//            CoreTypeConcrete::Coupon(_) => todo!(),
            CoreTypeConcrete::Coupon(_) => todo!(),
//        }
        }
//        .pad_to_align())
        .pad_to_align())
//    }
    }
//

//    fn is_memory_allocated(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool {
    fn is_memory_allocated(&self, registry: &ProgramRegistry<CoreType, CoreLibfunc>) -> bool {
//        // Right now, only enums and other structures which may end up passing a flattened enum as
        // Right now, only enums and other structures which may end up passing a flattened enum as
//        // arguments.
        // arguments.
//        match self {
        match self {
//            CoreTypeConcrete::Array(_) => false,
            CoreTypeConcrete::Array(_) => false,
//            CoreTypeConcrete::Bitwise(_) => false,
            CoreTypeConcrete::Bitwise(_) => false,
//            CoreTypeConcrete::Box(_) => false,
            CoreTypeConcrete::Box(_) => false,
//            CoreTypeConcrete::EcOp(_) => false,
            CoreTypeConcrete::EcOp(_) => false,
//            CoreTypeConcrete::EcPoint(_) => false,
            CoreTypeConcrete::EcPoint(_) => false,
//            CoreTypeConcrete::EcState(_) => false,
            CoreTypeConcrete::EcState(_) => false,
//            CoreTypeConcrete::Felt252(_) => false,
            CoreTypeConcrete::Felt252(_) => false,
//            CoreTypeConcrete::GasBuiltin(_) => false,
            CoreTypeConcrete::GasBuiltin(_) => false,
//            CoreTypeConcrete::BuiltinCosts(_) => false,
            CoreTypeConcrete::BuiltinCosts(_) => false,
//            CoreTypeConcrete::Uint8(_) => false,
            CoreTypeConcrete::Uint8(_) => false,
//            CoreTypeConcrete::Uint16(_) => false,
            CoreTypeConcrete::Uint16(_) => false,
//            CoreTypeConcrete::Uint32(_) => false,
            CoreTypeConcrete::Uint32(_) => false,
//            CoreTypeConcrete::Uint64(_) => false,
            CoreTypeConcrete::Uint64(_) => false,
//            CoreTypeConcrete::Uint128(_) => false,
            CoreTypeConcrete::Uint128(_) => false,
//            CoreTypeConcrete::Uint128MulGuarantee(_) => false,
            CoreTypeConcrete::Uint128MulGuarantee(_) => false,
//            CoreTypeConcrete::Sint8(_) => false,
            CoreTypeConcrete::Sint8(_) => false,
//            CoreTypeConcrete::Sint16(_) => false,
            CoreTypeConcrete::Sint16(_) => false,
//            CoreTypeConcrete::Sint32(_) => false,
            CoreTypeConcrete::Sint32(_) => false,
//            CoreTypeConcrete::Sint64(_) => false,
            CoreTypeConcrete::Sint64(_) => false,
//            CoreTypeConcrete::Sint128(_) => false,
            CoreTypeConcrete::Sint128(_) => false,
//            CoreTypeConcrete::NonZero(_) => false,
            CoreTypeConcrete::NonZero(_) => false,
//            CoreTypeConcrete::Nullable(_) => false,
            CoreTypeConcrete::Nullable(_) => false,
//            CoreTypeConcrete::RangeCheck(_) => false,
            CoreTypeConcrete::RangeCheck(_) => false,
//            CoreTypeConcrete::Uninitialized(_) => false,
            CoreTypeConcrete::Uninitialized(_) => false,
//            CoreTypeConcrete::Enum(info) => {
            CoreTypeConcrete::Enum(info) => {
//                // Enums are memory-allocated if either:
                // Enums are memory-allocated if either:
//                //   - Has only variant which is memory-allocated.
                //   - Has only variant which is memory-allocated.
//                //   - Has more than one variants, at least one of them being non-ZST.
                //   - Has more than one variants, at least one of them being non-ZST.
//                match info.variants.len() {
                match info.variants.len() {
//                    0 => false,
                    0 => false,
//                    1 => registry
                    1 => registry
//                        .get_type(&info.variants[0])
                        .get_type(&info.variants[0])
//                        .unwrap()
                        .unwrap()
//                        .is_memory_allocated(registry),
                        .is_memory_allocated(registry),
//                    _ => info
                    _ => info
//                        .variants
                        .variants
//                        .iter()
                        .iter()
//                        .any(|type_id| !registry.get_type(type_id).unwrap().is_zst(registry)),
                        .any(|type_id| !registry.get_type(type_id).unwrap().is_zst(registry)),
//                }
                }
//            }
            }
//            CoreTypeConcrete::Struct(info) => info.members.iter().any(|type_id| {
            CoreTypeConcrete::Struct(info) => info.members.iter().any(|type_id| {
//                // Structs are memory-allocated if any of its members is memory-allocated.
                // Structs are memory-allocated if any of its members is memory-allocated.
//                registry
                registry
//                    .get_type(type_id)
                    .get_type(type_id)
//                    .unwrap()
                    .unwrap()
//                    .is_memory_allocated(registry)
                    .is_memory_allocated(registry)
//            }),
            }),
//            CoreTypeConcrete::Felt252Dict(_) => false,
            CoreTypeConcrete::Felt252Dict(_) => false,
//            CoreTypeConcrete::Felt252DictEntry(_) => false,
            CoreTypeConcrete::Felt252DictEntry(_) => false,
//            CoreTypeConcrete::SquashedFelt252Dict(_) => false,
            CoreTypeConcrete::SquashedFelt252Dict(_) => false,
//            CoreTypeConcrete::Pedersen(_) => false,
            CoreTypeConcrete::Pedersen(_) => false,
//            CoreTypeConcrete::Poseidon(_) => false,
            CoreTypeConcrete::Poseidon(_) => false,
//            CoreTypeConcrete::Span(_) => false,
            CoreTypeConcrete::Span(_) => false,
//            CoreTypeConcrete::StarkNet(_) => false,
            CoreTypeConcrete::StarkNet(_) => false,
//            CoreTypeConcrete::SegmentArena(_) => false,
            CoreTypeConcrete::SegmentArena(_) => false,
//            CoreTypeConcrete::Snapshot(info) => registry
            CoreTypeConcrete::Snapshot(info) => registry
//                .get_type(&info.ty)
                .get_type(&info.ty)
//                .unwrap()
                .unwrap()
//                .is_memory_allocated(registry),
                .is_memory_allocated(registry),
//            CoreTypeConcrete::Bytes31(_) => false,
            CoreTypeConcrete::Bytes31(_) => false,
//

//            CoreTypeConcrete::BoundedInt(_) => false,
            CoreTypeConcrete::BoundedInt(_) => false,
//            CoreTypeConcrete::Const(info) => registry
            CoreTypeConcrete::Const(info) => registry
//                .get_type(&info.inner_ty)
                .get_type(&info.inner_ty)
//                .unwrap()
                .unwrap()
//                .is_memory_allocated(registry),
                .is_memory_allocated(registry),
//            CoreTypeConcrete::Coupon(_) => todo!(),
            CoreTypeConcrete::Coupon(_) => todo!(),
//        }
        }
//    }
    }
//

//    fn integer_width(&self) -> Option<usize> {
    fn integer_width(&self) -> Option<usize> {
//        match self {
        match self {
//            Self::Uint8(_) => Some(8),
            Self::Uint8(_) => Some(8),
//            Self::Uint16(_) => Some(16),
            Self::Uint16(_) => Some(16),
//            Self::Uint32(_) => Some(32),
            Self::Uint32(_) => Some(32),
//            Self::Uint64(_) => Some(64),
            Self::Uint64(_) => Some(64),
//            Self::Uint128(_) => Some(128),
            Self::Uint128(_) => Some(128),
//            Self::Felt252(_) => Some(252),
            Self::Felt252(_) => Some(252),
//            Self::Sint8(_) => Some(8),
            Self::Sint8(_) => Some(8),
//            Self::Sint16(_) => Some(16),
            Self::Sint16(_) => Some(16),
//            Self::Sint32(_) => Some(32),
            Self::Sint32(_) => Some(32),
//            Self::Sint64(_) => Some(64),
            Self::Sint64(_) => Some(64),
//            Self::Sint128(_) => Some(128),
            Self::Sint128(_) => Some(128),
//

//            CoreTypeConcrete::BoundedInt(_) => Some(252),
            CoreTypeConcrete::BoundedInt(_) => Some(252),
//            CoreTypeConcrete::Bytes31(_) => Some(248),
            CoreTypeConcrete::Bytes31(_) => Some(248),
//            CoreTypeConcrete::Const(_) => todo!(),
            CoreTypeConcrete::Const(_) => todo!(),
//

//            _ => None,
            _ => None,
//        }
        }
//    }
    }
//

//    fn is_integer_signed(&self) -> Option<bool> {
    fn is_integer_signed(&self) -> Option<bool> {
//        match self {
        match self {
//            Self::Uint8(_) => Some(false),
            Self::Uint8(_) => Some(false),
//            Self::Uint16(_) => Some(false),
            Self::Uint16(_) => Some(false),
//            Self::Uint32(_) => Some(false),
            Self::Uint32(_) => Some(false),
//            Self::Uint64(_) => Some(false),
            Self::Uint64(_) => Some(false),
//            Self::Uint128(_) => Some(false),
            Self::Uint128(_) => Some(false),
//            Self::Felt252(_) => Some(true),
            Self::Felt252(_) => Some(true),
//            Self::Sint8(_) => Some(true),
            Self::Sint8(_) => Some(true),
//            Self::Sint16(_) => Some(true),
            Self::Sint16(_) => Some(true),
//            Self::Sint32(_) => Some(true),
            Self::Sint32(_) => Some(true),
//            Self::Sint64(_) => Some(true),
            Self::Sint64(_) => Some(true),
//            Self::Sint128(_) => Some(true),
            Self::Sint128(_) => Some(true),
//

//            CoreTypeConcrete::BoundedInt(info) => {
            CoreTypeConcrete::BoundedInt(info) => {
//                Some(info.range.lower.is_negative() || info.range.upper.is_negative())
                Some(info.range.lower.is_negative() || info.range.upper.is_negative())
//            }
            }
//            CoreTypeConcrete::Bytes31(_) => Some(false),
            CoreTypeConcrete::Bytes31(_) => Some(false),
//            CoreTypeConcrete::Const(_) => todo!(),
            CoreTypeConcrete::Const(_) => todo!(),
//

//            _ => None,
            _ => None,
//        }
        }
//    }
    }
//

//    fn variants(&self) -> Option<&[ConcreteTypeId]> {
    fn variants(&self) -> Option<&[ConcreteTypeId]> {
//        match self {
        match self {
//            Self::Enum(info) => Some(&info.variants),
            Self::Enum(info) => Some(&info.variants),
//            _ => None,
            _ => None,
//        }
        }
//    }
    }
//

//    fn fields(&self) -> Option<&[ConcreteTypeId]> {
    fn fields(&self) -> Option<&[ConcreteTypeId]> {
//        match self {
        match self {
//            Self::Struct(info) => Some(&info.members),
            Self::Struct(info) => Some(&info.members),
//            _ => None,
            _ => None,
//        }
        }
//    }
    }
//

//    fn build_default<'ctx, 'this>(
    fn build_default<'ctx, 'this>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        entry: &'this Block<'ctx>,
        entry: &'this Block<'ctx>,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        _helper: &LibfuncHelper<'ctx, 'this>,
        _helper: &LibfuncHelper<'ctx, 'this>,
//        _metadata: &mut MetadataStorage,
        _metadata: &mut MetadataStorage,
//        _self_ty: &ConcreteTypeId,
        _self_ty: &ConcreteTypeId,
//    ) -> Result<Value<'ctx, 'this>, Self::Error> {
    ) -> Result<Value<'ctx, 'this>, Self::Error> {
//        static BOOL_USER_TYPE_ID: OnceLock<UserTypeId> = OnceLock::new();
        static BOOL_USER_TYPE_ID: OnceLock<UserTypeId> = OnceLock::new();
//        let bool_user_type_id =
        let bool_user_type_id =
//            BOOL_USER_TYPE_ID.get_or_init(|| UserTypeId::from_string("core::bool"));
            BOOL_USER_TYPE_ID.get_or_init(|| UserTypeId::from_string("core::bool"));
//

//        Ok(match self {
        Ok(match self {
//            Self::Enum(info) => match &info.info.long_id.generic_args[0] {
            Self::Enum(info) => match &info.info.long_id.generic_args[0] {
//                GenericArg::UserType(id) if id == bool_user_type_id => {
                GenericArg::UserType(id) if id == bool_user_type_id => {
//                    let tag = entry
                    let tag = entry
//                        .append_operation(arith::constant(
                        .append_operation(arith::constant(
//                            context,
                            context,
//                            IntegerAttribute::new(IntegerType::new(context, 1).into(), 0).into(),
                            IntegerAttribute::new(IntegerType::new(context, 1).into(), 0).into(),
//                            location,
                            location,
//                        ))
                        ))
//                        .result(0)?
                        .result(0)?
//                        .into();
                        .into();
//

//                    let value = entry
                    let value = entry
//                        .append_operation(llvm::undef(
                        .append_operation(llvm::undef(
//                            llvm::r#type::r#struct(
                            llvm::r#type::r#struct(
//                                context,
                                context,
//                                &[
                                &[
//                                    IntegerType::new(context, 1).into(),
                                    IntegerType::new(context, 1).into(),
//                                    llvm::r#type::array(IntegerType::new(context, 8).into(), 0),
                                    llvm::r#type::array(IntegerType::new(context, 8).into(), 0),
//                                ],
                                ],
//                                false,
                                false,
//                            ),
                            ),
//                            location,
                            location,
//                        ))
                        ))
//                        .result(0)?
                        .result(0)?
//                        .into();
                        .into();
//                    entry
                    entry
//                        .append_operation(llvm::insert_value(
                        .append_operation(llvm::insert_value(
//                            context,
                            context,
//                            value,
                            value,
//                            DenseI64ArrayAttribute::new(context, &[0]),
                            DenseI64ArrayAttribute::new(context, &[0]),
//                            tag,
                            tag,
//                            location,
                            location,
//                        ))
                        ))
//                        .result(0)?
                        .result(0)?
//                        .into()
                        .into()
//                }
                }
//                _ => unimplemented!("unsupported dict value type"),
                _ => unimplemented!("unsupported dict value type"),
//            },
            },
//            Self::Felt252(_) => entry
            Self::Felt252(_) => entry
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(IntegerType::new(context, 252).into(), 0).into(),
                    IntegerAttribute::new(IntegerType::new(context, 252).into(), 0).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into(),
                .into(),
//            Self::Nullable(_) => entry
            Self::Nullable(_) => entry
//                .append_operation(
                .append_operation(
//                    ods::llvm::mlir_zero(context, pointer(context, 0), location).into(),
                    ods::llvm::mlir_zero(context, pointer(context, 0), location).into(),
//                )
                )
//                .result(0)?
                .result(0)?
//                .into(),
                .into(),
//            Self::Uint8(_) => entry
            Self::Uint8(_) => entry
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(IntegerType::new(context, 8).into(), 0).into(),
                    IntegerAttribute::new(IntegerType::new(context, 8).into(), 0).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into(),
                .into(),
//            Self::Uint16(_) => entry
            Self::Uint16(_) => entry
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(IntegerType::new(context, 16).into(), 0).into(),
                    IntegerAttribute::new(IntegerType::new(context, 16).into(), 0).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into(),
                .into(),
//            Self::Uint32(_) => entry
            Self::Uint32(_) => entry
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(IntegerType::new(context, 32).into(), 0).into(),
                    IntegerAttribute::new(IntegerType::new(context, 32).into(), 0).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into(),
                .into(),
//            Self::Uint64(_) => entry
            Self::Uint64(_) => entry
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
                    IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into(),
                .into(),
//            Self::Uint128(_) => entry
            Self::Uint128(_) => entry
//                .append_operation(arith::constant(
                .append_operation(arith::constant(
//                    context,
                    context,
//                    IntegerAttribute::new(IntegerType::new(context, 128).into(), 0).into(),
                    IntegerAttribute::new(IntegerType::new(context, 128).into(), 0).into(),
//                    location,
                    location,
//                ))
                ))
//                .result(0)?
                .result(0)?
//                .into(),
                .into(),
//            _ => unimplemented!("unsupported dict value type"),
            _ => unimplemented!("unsupported dict value type"),
//        })
        })
//    }
    }
//

//    fn build_drop<'ctx, 'this>(
    fn build_drop<'ctx, 'this>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        entry: &'this Block<'ctx>,
        entry: &'this Block<'ctx>,
//        location: Location<'ctx>,
        location: Location<'ctx>,
//        helper: &LibfuncHelper<'ctx, 'this>,
        helper: &LibfuncHelper<'ctx, 'this>,
//        metadata: &mut MetadataStorage,
        metadata: &mut MetadataStorage,
//        self_ty: &ConcreteTypeId,
        self_ty: &ConcreteTypeId,
//    ) -> Result<(), Self::Error> {
    ) -> Result<(), Self::Error> {
//        match self {
        match self {
//            CoreTypeConcrete::Array(_info) => {
            CoreTypeConcrete::Array(_info) => {
//                if metadata.get::<ReallocBindingsMeta>().is_none() {
                if metadata.get::<ReallocBindingsMeta>().is_none() {
//                    metadata.insert(ReallocBindingsMeta::new(context, helper));
                    metadata.insert(ReallocBindingsMeta::new(context, helper));
//                }
                }
//

//                let array_ty = registry.build_type(context, helper, registry, metadata, self_ty)?;
                let array_ty = registry.build_type(context, helper, registry, metadata, self_ty)?;
//

//                let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
                let ptr_ty = crate::ffi::get_struct_field_type_at(&array_ty, 0);
//

//                let array_val = entry.argument(0)?.into();
                let array_val = entry.argument(0)?.into();
//

//                let op = entry.append_operation(llvm::extract_value(
                let op = entry.append_operation(llvm::extract_value(
//                    context,
                    context,
//                    array_val,
                    array_val,
//                    DenseI64ArrayAttribute::new(context, &[0]),
                    DenseI64ArrayAttribute::new(context, &[0]),
//                    ptr_ty,
                    ptr_ty,
//                    location,
                    location,
//                ));
                ));
//                let ptr: Value = op.result(0)?.into();
                let ptr: Value = op.result(0)?.into();
//

//                entry.append_operation(ReallocBindingsMeta::free(context, ptr, location));
                entry.append_operation(ReallocBindingsMeta::free(context, ptr, location));
//            }
            }
//            CoreTypeConcrete::Felt252Dict(_) | CoreTypeConcrete::SquashedFelt252Dict(_) => {
            CoreTypeConcrete::Felt252Dict(_) | CoreTypeConcrete::SquashedFelt252Dict(_) => {
//                let runtime: &mut RuntimeBindingsMeta = metadata.get_mut().unwrap();
                let runtime: &mut RuntimeBindingsMeta = metadata.get_mut().unwrap();
//                let ptr = entry.argument(0)?.into();
                let ptr = entry.argument(0)?.into();
//

//                runtime.dict_alloc_free(context, helper, ptr, entry, location)?;
                runtime.dict_alloc_free(context, helper, ptr, entry, location)?;
//            }
            }
//            CoreTypeConcrete::Box(_) | CoreTypeConcrete::Nullable(_) => {
            CoreTypeConcrete::Box(_) | CoreTypeConcrete::Nullable(_) => {
//                if metadata.get::<ReallocBindingsMeta>().is_none() {
                if metadata.get::<ReallocBindingsMeta>().is_none() {
//                    metadata.insert(ReallocBindingsMeta::new(context, helper));
                    metadata.insert(ReallocBindingsMeta::new(context, helper));
//                }
                }
//

//                let ptr = entry.argument(0)?.into();
                let ptr = entry.argument(0)?.into();
//                entry.append_operation(ReallocBindingsMeta::free(context, ptr, location));
                entry.append_operation(ReallocBindingsMeta::free(context, ptr, location));
//            }
            }
//            _ => {}
            _ => {}
//        };
        };
//        Ok(())
        Ok(())
//    }
    }
//}
}
//

//#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
//pub struct WithSelf<'a, T> {
pub struct WithSelf<'a, T> {
//    self_ty: &'a ConcreteTypeId,
    self_ty: &'a ConcreteTypeId,
//    inner: &'a T,
    inner: &'a T,
//}
}
//

//impl<'a, T> WithSelf<'a, T> {
impl<'a, T> WithSelf<'a, T> {
//    pub fn new(self_ty: &'a ConcreteTypeId, inner: &'a T) -> Self {
    pub fn new(self_ty: &'a ConcreteTypeId, inner: &'a T) -> Self {
//        Self { self_ty, inner }
        Self { self_ty, inner }
//    }
    }
//

//    pub fn self_ty(&self) -> &ConcreteTypeId {
    pub fn self_ty(&self) -> &ConcreteTypeId {
//        self.self_ty
        self.self_ty
//    }
    }
//}
}
//

//impl<'a, T> AsRef<T> for WithSelf<'a, T> {
impl<'a, T> AsRef<T> for WithSelf<'a, T> {
//    fn as_ref(&self) -> &T {
    fn as_ref(&self) -> &T {
//        self.inner
        self.inner
//    }
    }
//}
}
//

//impl<'a, T> Deref for WithSelf<'a, T> {
impl<'a, T> Deref for WithSelf<'a, T> {
//    type Target = T;
    type Target = T;
//

//    fn deref(&self) -> &T {
    fn deref(&self) -> &T {
//        self.inner
        self.inner
//    }
    }
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use super::TypeBuilder;
    use super::TypeBuilder;
//    use crate::utils::test::load_cairo;
    use crate::utils::test::load_cairo;
//    use cairo_lang_sierra::{
    use cairo_lang_sierra::{
//        extensions::core::{CoreLibfunc, CoreType},
        extensions::core::{CoreLibfunc, CoreType},
//        program_registry::ProgramRegistry,
        program_registry::ProgramRegistry,
//    };
    };
//

//    #[test]
    #[test]
//    fn ensure_padded_layouts() {
    fn ensure_padded_layouts() {
//        let (_, program) = load_cairo! {
        let (_, program) = load_cairo! {
//            #[derive(Drop)]
            #[derive(Drop)]
//            struct A {}
            struct A {}
//            #[derive(Drop)]
            #[derive(Drop)]
//            struct B { a: u8 }
            struct B { a: u8 }
//            #[derive(Drop)]
            #[derive(Drop)]
//            struct C { a: u8, b: u16 }
            struct C { a: u8, b: u16 }
//            #[derive(Drop)]
            #[derive(Drop)]
//            struct D { a: u16, b: u8 }
            struct D { a: u16, b: u8 }
//

//            fn main(a: A, b: B, c: C, d: D) {}
            fn main(a: A, b: B, c: C, d: D) {}
//        };
        };
//

//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//        for ty in &program.type_declarations {
        for ty in &program.type_declarations {
//            let ty = registry.get_type(&ty.id).unwrap();
            let ty = registry.get_type(&ty.id).unwrap();
//            let layout = ty.layout(&registry).unwrap();
            let layout = ty.layout(&registry).unwrap();
//            assert_eq!(layout, layout.pad_to_align());
            assert_eq!(layout, layout.pad_to_align());
//        }
        }
//    }
    }
//}
}
