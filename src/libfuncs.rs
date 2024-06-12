////! # Compiler libfunc infrastructure
//! # Compiler libfunc infrastructure
////!
//!
////! Contains libfunc generation stuff (aka. the actual instructions).
//! Contains libfunc generation stuff (aka. the actual instructions).
//

//use crate::{error::Error as CoreLibfuncBuilderError, metadata::MetadataStorage};
use crate::{error::Error as CoreLibfuncBuilderError, metadata::MetadataStorage};
//use bumpalo::Bump;
use bumpalo::Bump;
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::core::{CoreConcreteLibfunc, CoreLibfunc, CoreType},
    extensions::core::{CoreConcreteLibfunc, CoreLibfunc, CoreType},
//    ids::FunctionId,
    ids::FunctionId,
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::{arith, cf},
    dialect::{arith, cf},
//    ir::{
    ir::{
//        attribute::IntegerAttribute, r#type::IntegerType, Block, BlockRef, Location, Module,
        attribute::IntegerAttribute, r#type::IntegerType, Block, BlockRef, Location, Module,
//        Operation, Region, Value, ValueLike,
        Operation, Region, Value, ValueLike,
//    },
    },
//    Context,
    Context,
//};
};
//use std::{borrow::Cow, cell::Cell, error::Error, ops::Deref};
use std::{borrow::Cow, cell::Cell, error::Error, ops::Deref};
//

//pub mod ap_tracking;
pub mod ap_tracking;
//pub mod array;
pub mod array;
//pub mod bitwise;
pub mod bitwise;
//pub mod r#bool;
pub mod r#bool;
//pub mod r#box;
pub mod r#box;
//pub mod branch_align;
pub mod branch_align;
//pub mod bytes31;
pub mod bytes31;
//pub mod cast;
pub mod cast;
//pub mod const_libfunc;
pub mod const_libfunc;
//pub mod debug;
pub mod debug;
//pub mod drop;
pub mod drop;
//pub mod dup;
pub mod dup;
//pub mod ec;
pub mod ec;
//pub mod r#enum;
pub mod r#enum;
//pub mod felt252;
pub mod felt252;
//pub mod felt252_dict;
pub mod felt252_dict;
//pub mod felt252_dict_entry;
pub mod felt252_dict_entry;
//pub mod function_call;
pub mod function_call;
//pub mod gas;
pub mod gas;
//pub mod mem;
pub mod mem;
//pub mod nullable;
pub mod nullable;
//pub mod pedersen;
pub mod pedersen;
//pub mod poseidon;
pub mod poseidon;
//pub mod sint128;
pub mod sint128;
//pub mod sint16;
pub mod sint16;
//pub mod sint32;
pub mod sint32;
//pub mod sint64;
pub mod sint64;
//pub mod sint8;
pub mod sint8;
//pub mod snapshot_take;
pub mod snapshot_take;
//pub mod starknet;
pub mod starknet;
//pub mod r#struct;
pub mod r#struct;
//pub mod uint128;
pub mod uint128;
//pub mod uint16;
pub mod uint16;
//pub mod uint256;
pub mod uint256;
//pub mod uint32;
pub mod uint32;
//pub mod uint512;
pub mod uint512;
//pub mod uint64;
pub mod uint64;
//pub mod uint8;
pub mod uint8;
//pub mod unconditional_jump;
pub mod unconditional_jump;
//pub mod unwrap_non_zero;
pub mod unwrap_non_zero;
//

///// Generation of MLIR operations from their Sierra counterparts.
/// Generation of MLIR operations from their Sierra counterparts.
/////
///
///// All possible Sierra libfuncs must implement it. It is already implemented for all the core
/// All possible Sierra libfuncs must implement it. It is already implemented for all the core
///// libfuncs, contained in [CoreConcreteLibfunc].
/// libfuncs, contained in [CoreConcreteLibfunc].
//pub trait LibfuncBuilder {
pub trait LibfuncBuilder {
//    /// Error type returned by this trait's methods.
    /// Error type returned by this trait's methods.
//    type Error: Error;
    type Error: Error;
//

//    /// Generate the MLIR operations.
    /// Generate the MLIR operations.
//    fn build<'ctx, 'this>(
    fn build<'ctx, 'this>(
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
//    ) -> Result<(), Self::Error>;
    ) -> Result<(), Self::Error>;
//

//    /// Return the target function if the statement is a function call.
    /// Return the target function if the statement is a function call.
//    ///
    ///
//    /// This is used by the compiler to check whether a statement is a function call and apply the
    /// This is used by the compiler to check whether a statement is a function call and apply the
//    /// tail recursion logic.
    /// tail recursion logic.
//    fn is_function_call(&self) -> Option<&FunctionId>;
    fn is_function_call(&self) -> Option<&FunctionId>;
//}
}
//

//impl LibfuncBuilder for CoreConcreteLibfunc {
impl LibfuncBuilder for CoreConcreteLibfunc {
//    type Error = CoreLibfuncBuilderError;
    type Error = CoreLibfuncBuilderError;
//

//    fn build<'ctx, 'this>(
    fn build<'ctx, 'this>(
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
//    ) -> Result<(), Self::Error> {
    ) -> Result<(), Self::Error> {
//        match self {
        match self {
//            Self::ApTracking(selector) => self::ap_tracking::build(
            Self::ApTracking(selector) => self::ap_tracking::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Array(selector) => self::array::build(
            Self::Array(selector) => self::array::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::BranchAlign(info) => self::branch_align::build(
            Self::BranchAlign(info) => self::branch_align::build(
//                context, registry, entry, location, helper, metadata, info,
                context, registry, entry, location, helper, metadata, info,
//            ),
            ),
//            Self::Bool(selector) => self::r#bool::build(
            Self::Bool(selector) => self::r#bool::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Box(selector) => self::r#box::build(
            Self::Box(selector) => self::r#box::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Bytes31(selector) => self::bytes31::build(
            Self::Bytes31(selector) => self::bytes31::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Cast(selector) => self::cast::build(
            Self::Cast(selector) => self::cast::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Const(selector) => self::const_libfunc::build(
            Self::Const(selector) => self::const_libfunc::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Debug(selector) => self::debug::build(
            Self::Debug(selector) => self::debug::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Drop(info) => {
            Self::Drop(info) => {
//                self::drop::build(context, registry, entry, location, helper, metadata, info)
                self::drop::build(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Self::Dup(info) => {
            Self::Dup(info) => {
//                self::dup::build(context, registry, entry, location, helper, metadata, info)
                self::dup::build(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Self::Ec(selector) => self::ec::build(
            Self::Ec(selector) => self::ec::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Enum(selector) => self::r#enum::build(
            Self::Enum(selector) => self::r#enum::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Felt252(selector) => self::felt252::build(
            Self::Felt252(selector) => self::felt252::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Felt252Dict(selector) => self::felt252_dict::build(
            Self::Felt252Dict(selector) => self::felt252_dict::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Felt252DictEntry(selector) => self::felt252_dict_entry::build(
            Self::Felt252DictEntry(selector) => self::felt252_dict_entry::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::FunctionCall(info) => self::function_call::build(
            Self::FunctionCall(info) => self::function_call::build(
//                context, registry, entry, location, helper, metadata, info,
                context, registry, entry, location, helper, metadata, info,
//            ),
            ),
//            Self::Gas(selector) => self::gas::build(
            Self::Gas(selector) => self::gas::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Mem(selector) => self::mem::build(
            Self::Mem(selector) => self::mem::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Nullable(selector) => self::nullable::build(
            Self::Nullable(selector) => self::nullable::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Pedersen(selector) => self::pedersen::build(
            Self::Pedersen(selector) => self::pedersen::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Poseidon(selector) => self::poseidon::build(
            Self::Poseidon(selector) => self::poseidon::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Sint8(info) => {
            Self::Sint8(info) => {
//                self::sint8::build(context, registry, entry, location, helper, metadata, info)
                self::sint8::build(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Self::Sint16(info) => {
            Self::Sint16(info) => {
//                self::sint16::build(context, registry, entry, location, helper, metadata, info)
                self::sint16::build(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Self::Sint32(info) => {
            Self::Sint32(info) => {
//                self::sint32::build(context, registry, entry, location, helper, metadata, info)
                self::sint32::build(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Self::Sint64(info) => {
            Self::Sint64(info) => {
//                self::sint64::build(context, registry, entry, location, helper, metadata, info)
                self::sint64::build(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Self::Sint128(info) => {
            Self::Sint128(info) => {
//                self::sint128::build(context, registry, entry, location, helper, metadata, info)
                self::sint128::build(context, registry, entry, location, helper, metadata, info)
//            }
            }
//            Self::SnapshotTake(info) => self::snapshot_take::build(
            Self::SnapshotTake(info) => self::snapshot_take::build(
//                context, registry, entry, location, helper, metadata, info,
                context, registry, entry, location, helper, metadata, info,
//            ),
            ),
//            Self::StarkNet(selector) => self::starknet::build(
            Self::StarkNet(selector) => self::starknet::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Struct(selector) => self::r#struct::build(
            Self::Struct(selector) => self::r#struct::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Uint8(selector) => self::uint8::build(
            Self::Uint8(selector) => self::uint8::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Uint16(selector) => self::uint16::build(
            Self::Uint16(selector) => self::uint16::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Uint32(selector) => self::uint32::build(
            Self::Uint32(selector) => self::uint32::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Uint64(selector) => self::uint64::build(
            Self::Uint64(selector) => self::uint64::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Uint128(selector) => self::uint128::build(
            Self::Uint128(selector) => self::uint128::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Uint256(selector) => self::uint256::build(
            Self::Uint256(selector) => self::uint256::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::Uint512(selector) => self::uint512::build(
            Self::Uint512(selector) => self::uint512::build(
//                context, registry, entry, location, helper, metadata, selector,
                context, registry, entry, location, helper, metadata, selector,
//            ),
            ),
//            Self::UnconditionalJump(info) => self::unconditional_jump::build(
            Self::UnconditionalJump(info) => self::unconditional_jump::build(
//                context, registry, entry, location, helper, metadata, info,
                context, registry, entry, location, helper, metadata, info,
//            ),
            ),
//            Self::UnwrapNonZero(info) => self::unwrap_non_zero::build(
            Self::UnwrapNonZero(info) => self::unwrap_non_zero::build(
//                context, registry, entry, location, helper, metadata, info,
                context, registry, entry, location, helper, metadata, info,
//            ),
            ),
//            Self::Coupon(_) => todo!(),
            Self::Coupon(_) => todo!(),
//            Self::CouponCall(_) => todo!(),
            Self::CouponCall(_) => todo!(),
//        }
        }
//    }
    }
//

//    fn is_function_call(&self) -> Option<&FunctionId> {
    fn is_function_call(&self) -> Option<&FunctionId> {
//        match self {
        match self {
//            CoreConcreteLibfunc::FunctionCall(info) => Some(&info.function.id),
            CoreConcreteLibfunc::FunctionCall(info) => Some(&info.function.id),
//            _ => None,
            _ => None,
//        }
        }
//    }
    }
//}
}
//

///// Helper struct which contains logic generation for extra MLIR blocks and branch operations to the
/// Helper struct which contains logic generation for extra MLIR blocks and branch operations to the
///// next statements.
/// next statements.
/////
///
///// Each branch index should be present in exactly one call a branching method (either
/// Each branch index should be present in exactly one call a branching method (either
///// [`br`](#method.br), [`cond_br`](#method.cond_br) or [`switch`](#method.switch)).
/// [`br`](#method.br), [`cond_br`](#method.cond_br) or [`switch`](#method.switch)).
/////
///
///// This helper is necessary because the statement following the current one may not have the same
/// This helper is necessary because the statement following the current one may not have the same
///// arguments as the results returned by the current statement. Because of that, a direct jump
/// arguments as the results returned by the current statement. Because of that, a direct jump
///// cannot be made and some processing is required.
/// cannot be made and some processing is required.
//pub struct LibfuncHelper<'ctx, 'this>
pub struct LibfuncHelper<'ctx, 'this>
//where
where
//    'this: 'ctx,
    'this: 'ctx,
//{
{
//    pub(crate) module: &'this Module<'ctx>,
    pub(crate) module: &'this Module<'ctx>,
//    pub(crate) init_block: &'this BlockRef<'ctx, 'this>,
    pub(crate) init_block: &'this BlockRef<'ctx, 'this>,
//

//    pub(crate) region: &'this Region<'ctx>,
    pub(crate) region: &'this Region<'ctx>,
//    pub(crate) blocks_arena: &'this Bump,
    pub(crate) blocks_arena: &'this Bump,
//    pub(crate) last_block: Cell<&'this BlockRef<'ctx, 'this>>,
    pub(crate) last_block: Cell<&'this BlockRef<'ctx, 'this>>,
//

//    pub(crate) branches: Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>,
    pub(crate) branches: Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>,
//    pub(crate) results: Vec<Vec<Cell<Option<Value<'ctx, 'this>>>>>,
    pub(crate) results: Vec<Vec<Cell<Option<Value<'ctx, 'this>>>>>,
//}
}
//

//impl<'ctx, 'this> LibfuncHelper<'ctx, 'this>
impl<'ctx, 'this> LibfuncHelper<'ctx, 'this>
//where
where
//    'this: 'ctx,
    'this: 'ctx,
//{
{
//    pub(crate) fn results(self) -> impl Iterator<Item = Vec<Value<'ctx, 'this>>> {
    pub(crate) fn results(self) -> impl Iterator<Item = Vec<Value<'ctx, 'this>>> {
//        self.results.into_iter().enumerate().map(|(branch_idx, x)| {
        self.results.into_iter().enumerate().map(|(branch_idx, x)| {
//            x.into_iter()
            x.into_iter()
//                .enumerate()
                .enumerate()
//                .map(|(arg_idx, x)| {
                .map(|(arg_idx, x)| {
//                    x.into_inner().unwrap_or_else(|| {
                    x.into_inner().unwrap_or_else(|| {
//                        panic!("Argument #{arg_idx} of branch {branch_idx} doesn't have a value.")
                        panic!("Argument #{arg_idx} of branch {branch_idx} doesn't have a value.")
//                    })
                    })
//                })
                })
//                .collect()
                .collect()
//        })
        })
//    }
    }
//

//    /// Return the initialization block.
    /// Return the initialization block.
//    ///
    ///
//    /// The init block is used for `llvm.alloca` instructions. It is guaranteed to not be executed
    /// The init block is used for `llvm.alloca` instructions. It is guaranteed to not be executed
//    /// multiple times on tail-recursive functions. This property allows generating tail-recursive
    /// multiple times on tail-recursive functions. This property allows generating tail-recursive
//    /// functions that do not grow the stack.
    /// functions that do not grow the stack.
//    pub fn init_block(&self) -> &'this Block<'ctx> {
    pub fn init_block(&self) -> &'this Block<'ctx> {
//        self.init_block
        self.init_block
//    }
    }
//

//    /// Inserts a new block after all the current libfunc's blocks.
    /// Inserts a new block after all the current libfunc's blocks.
//    pub fn append_block(&self, block: Block<'ctx>) -> &'this Block<'ctx> {
    pub fn append_block(&self, block: Block<'ctx>) -> &'this Block<'ctx> {
//        let block = self
        let block = self
//            .region
            .region
//            .insert_block_after(*self.last_block.get(), block);
            .insert_block_after(*self.last_block.get(), block);
//

//        let block_ref: &'this mut BlockRef<'ctx, 'this> = self.blocks_arena.alloc(block);
        let block_ref: &'this mut BlockRef<'ctx, 'this> = self.blocks_arena.alloc(block);
//        self.last_block.set(block_ref);
        self.last_block.set(block_ref);
//

//        block_ref
        block_ref
//    }
    }
//

//    /// Creates an unconditional branching operation out of the libfunc and into the next statement.
    /// Creates an unconditional branching operation out of the libfunc and into the next statement.
//    ///
    ///
//    /// This method will also store the returned values so that they can be moved into the state and
    /// This method will also store the returned values so that they can be moved into the state and
//    /// used later on when required.
    /// used later on when required.
//    pub fn br(
    pub fn br(
//        &self,
        &self,
//        branch: usize,
        branch: usize,
//        results: &[Value<'ctx, 'this>],
        results: &[Value<'ctx, 'this>],
//        location: Location<'ctx>,
        location: Location<'ctx>,
//    ) -> Operation<'ctx> {
    ) -> Operation<'ctx> {
//        let (successor, operands) = &self.branches[branch];
        let (successor, operands) = &self.branches[branch];
//

//        for (dst, src) in self.results[branch].iter().zip(results) {
        for (dst, src) in self.results[branch].iter().zip(results) {
//            dst.replace(Some(*src));
            dst.replace(Some(*src));
//        }
        }
//

//        let destination_operands = operands
        let destination_operands = operands
//            .iter()
            .iter()
//            .copied()
            .copied()
//            .map(|op| match op {
            .map(|op| match op {
//                BranchArg::External(x) => x,
                BranchArg::External(x) => x,
//                BranchArg::Returned(i) => results[i],
                BranchArg::Returned(i) => results[i],
//            })
            })
//            .collect::<Vec<_>>();
            .collect::<Vec<_>>();
//

//        cf::br(successor, &destination_operands, location)
        cf::br(successor, &destination_operands, location)
//    }
    }
//

//    /// Creates a conditional binary branching operation, potentially jumping out of the libfunc and
    /// Creates a conditional binary branching operation, potentially jumping out of the libfunc and
//    /// into the next statement.
    /// into the next statement.
//    ///
    ///
//    /// While generating a `cond_br` that doesn't jump out of the libfunc is possible, it should be
    /// While generating a `cond_br` that doesn't jump out of the libfunc is possible, it should be
//    /// avoided whenever possible. In those cases just use [melior::dialect::cf::cond_br].
    /// avoided whenever possible. In those cases just use [melior::dialect::cf::cond_br].
//    ///
    ///
//    /// This method will also store the returned values so that they can be moved into the state and
    /// This method will also store the returned values so that they can be moved into the state and
//    /// used later on when required.
    /// used later on when required.
//    // TODO: Allow one block to be libfunc-internal.
    // TODO: Allow one block to be libfunc-internal.
//    pub fn cond_br(
    pub fn cond_br(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        condition: Value<'ctx, 'this>,
        condition: Value<'ctx, 'this>,
//        branches: [usize; 2],
        branches: [usize; 2],
//        results: [&[Value<'ctx, 'this>]; 2],
        results: [&[Value<'ctx, 'this>]; 2],
//        location: Location<'ctx>,
        location: Location<'ctx>,
//    ) -> Operation<'ctx> {
    ) -> Operation<'ctx> {
//        let (block_true, args_true) = {
        let (block_true, args_true) = {
//            let (successor, operands) = &self.branches[branches[0]];
            let (successor, operands) = &self.branches[branches[0]];
//

//            for (dst, src) in self.results[branches[0]].iter().zip(results[0]) {
            for (dst, src) in self.results[branches[0]].iter().zip(results[0]) {
//                dst.replace(Some(*src));
                dst.replace(Some(*src));
//            }
            }
//

//            let destination_operands = operands
            let destination_operands = operands
//                .iter()
                .iter()
//                .copied()
                .copied()
//                .map(|op| match op {
                .map(|op| match op {
//                    BranchArg::External(x) => x,
                    BranchArg::External(x) => x,
//                    BranchArg::Returned(i) => results[0][i],
                    BranchArg::Returned(i) => results[0][i],
//                })
                })
//                .collect::<Vec<_>>();
                .collect::<Vec<_>>();
//

//            (*successor, destination_operands)
            (*successor, destination_operands)
//        };
        };
//

//        let (block_false, args_false) = {
        let (block_false, args_false) = {
//            let (successor, operands) = &self.branches[branches[1]];
            let (successor, operands) = &self.branches[branches[1]];
//

//            for (dst, src) in self.results[branches[1]].iter().zip(results[1]) {
            for (dst, src) in self.results[branches[1]].iter().zip(results[1]) {
//                dst.replace(Some(*src));
                dst.replace(Some(*src));
//            }
            }
//

//            let destination_operands = operands
            let destination_operands = operands
//                .iter()
                .iter()
//                .copied()
                .copied()
//                .map(|op| match op {
                .map(|op| match op {
//                    BranchArg::External(x) => x,
                    BranchArg::External(x) => x,
//                    BranchArg::Returned(i) => results[1][i],
                    BranchArg::Returned(i) => results[1][i],
//                })
                })
//                .collect::<Vec<_>>();
                .collect::<Vec<_>>();
//

//            (*successor, destination_operands)
            (*successor, destination_operands)
//        };
        };
//

//        cf::cond_br(
        cf::cond_br(
//            context,
            context,
//            condition,
            condition,
//            block_true,
            block_true,
//            block_false,
            block_false,
//            &args_true,
            &args_true,
//            &args_false,
            &args_false,
//            location,
            location,
//        )
        )
//    }
    }
//

//    /// Creates a conditional multi-branching operation, potentially jumping out of the libfunc and
    /// Creates a conditional multi-branching operation, potentially jumping out of the libfunc and
//    /// into the next statement.
    /// into the next statement.
//    ///
    ///
//    /// While generating a `switch` that doesn't jump out of the libfunc is possible, it should be
    /// While generating a `switch` that doesn't jump out of the libfunc is possible, it should be
//    /// avoided whenever possible. In those cases just use [melior::dialect::cf::switch].
    /// avoided whenever possible. In those cases just use [melior::dialect::cf::switch].
//    ///
    ///
//    /// This method will also store the returned values so that they can be moved into the state and
    /// This method will also store the returned values so that they can be moved into the state and
//    /// used later on when required.
    /// used later on when required.
//    pub fn switch(
    pub fn switch(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        flag: Value<'ctx, 'this>,
        flag: Value<'ctx, 'this>,
//        default: (BranchTarget<'ctx, '_>, &[Value<'ctx, 'this>]),
        default: (BranchTarget<'ctx, '_>, &[Value<'ctx, 'this>]),
//        branches: &[(i64, BranchTarget<'ctx, '_>, &[Value<'ctx, 'this>])],
        branches: &[(i64, BranchTarget<'ctx, '_>, &[Value<'ctx, 'this>])],
//        location: Location<'ctx>,
        location: Location<'ctx>,
//    ) -> Result<Operation<'ctx>, CoreLibfuncBuilderError> {
    ) -> Result<Operation<'ctx>, CoreLibfuncBuilderError> {
//        let default_destination = match default.0 {
        let default_destination = match default.0 {
//            BranchTarget::Jump(x) => (x, Cow::Borrowed(default.1)),
            BranchTarget::Jump(x) => (x, Cow::Borrowed(default.1)),
//            BranchTarget::Return(i) => {
            BranchTarget::Return(i) => {
//                let (successor, operands) = &self.branches[i];
                let (successor, operands) = &self.branches[i];
//

//                for (dst, src) in self.results[i].iter().zip(default.1) {
                for (dst, src) in self.results[i].iter().zip(default.1) {
//                    dst.replace(Some(*src));
                    dst.replace(Some(*src));
//                }
                }
//

//                let destination_operands = operands
                let destination_operands = operands
//                    .iter()
                    .iter()
//                    .copied()
                    .copied()
//                    .map(|op| match op {
                    .map(|op| match op {
//                        BranchArg::External(x) => x,
                        BranchArg::External(x) => x,
//                        BranchArg::Returned(i) => default.1[i],
                        BranchArg::Returned(i) => default.1[i],
//                    })
                    })
//                    .collect();
                    .collect();
//

//                (*successor, Cow::Owned(destination_operands))
                (*successor, Cow::Owned(destination_operands))
//            }
            }
//        };
        };
//

//        let mut case_values = Vec::with_capacity(branches.len());
        let mut case_values = Vec::with_capacity(branches.len());
//        let mut case_destinations = Vec::with_capacity(branches.len());
        let mut case_destinations = Vec::with_capacity(branches.len());
//        for (flag, successor, operands) in branches {
        for (flag, successor, operands) in branches {
//            case_values.push(*flag);
            case_values.push(*flag);
//

//            case_destinations.push(match *successor {
            case_destinations.push(match *successor {
//                BranchTarget::Jump(x) => (x, Cow::Borrowed(*operands)),
                BranchTarget::Jump(x) => (x, Cow::Borrowed(*operands)),
//                BranchTarget::Return(i) => {
                BranchTarget::Return(i) => {
//                    let (successor, operands) = &self.branches[i];
                    let (successor, operands) = &self.branches[i];
//

//                    for (dst, src) in self.results[i].iter().zip(default.1) {
                    for (dst, src) in self.results[i].iter().zip(default.1) {
//                        dst.replace(Some(*src));
                        dst.replace(Some(*src));
//                    }
                    }
//

//                    let destination_operands = operands
                    let destination_operands = operands
//                        .iter()
                        .iter()
//                        .copied()
                        .copied()
//                        .map(|op| match op {
                        .map(|op| match op {
//                            BranchArg::External(x) => x,
                            BranchArg::External(x) => x,
//                            BranchArg::Returned(i) => default.1[i],
                            BranchArg::Returned(i) => default.1[i],
//                        })
                        })
//                        .collect();
                        .collect();
//

//                    (*successor, Cow::Owned(destination_operands))
                    (*successor, Cow::Owned(destination_operands))
//                }
                }
//            });
            });
//        }
        }
//

//        Ok(cf::switch(
        Ok(cf::switch(
//            context,
            context,
//            &case_values,
            &case_values,
//            flag,
            flag,
//            flag.r#type(),
            flag.r#type(),
//            (default_destination.0, &default_destination.1),
            (default_destination.0, &default_destination.1),
//            &case_destinations
            &case_destinations
//                .iter()
                .iter()
//                .map(|(x, y)| (*x, y.as_ref()))
                .map(|(x, y)| (*x, y.as_ref()))
//                .collect::<Vec<_>>(),
                .collect::<Vec<_>>(),
//            location,
            location,
//        )?)
        )?)
//    }
    }
//}
}
//

//impl<'ctx, 'this> Deref for LibfuncHelper<'ctx, 'this> {
impl<'ctx, 'this> Deref for LibfuncHelper<'ctx, 'this> {
//    type Target = Module<'ctx>;
    type Target = Module<'ctx>;
//

//    fn deref(&self) -> &Self::Target {
    fn deref(&self) -> &Self::Target {
//        self.module
        self.module
//    }
    }
//}
}
//

//#[derive(Clone, Copy, Debug)]
#[derive(Clone, Copy, Debug)]
//pub(crate) enum BranchArg<'ctx, 'this> {
pub(crate) enum BranchArg<'ctx, 'this> {
//    External(Value<'ctx, 'this>),
    External(Value<'ctx, 'this>),
//    Returned(usize),
    Returned(usize),
//}
}
//

///// A libfunc branching target.
/// A libfunc branching target.
/////
///
///// May point to either a block within the same libfunc using [BranchTarget::Jump] or to one of the
/// May point to either a block within the same libfunc using [BranchTarget::Jump] or to one of the
///// statement's branches using [BranchTarget::Return] with the branch index.
/// statement's branches using [BranchTarget::Return] with the branch index.
//#[derive(Clone, Copy, Debug)]
#[derive(Clone, Copy, Debug)]
//pub enum BranchTarget<'ctx, 'a> {
pub enum BranchTarget<'ctx, 'a> {
//    /// A block within the current libfunc.
    /// A block within the current libfunc.
//    Jump(&'a Block<'ctx>),
    Jump(&'a Block<'ctx>),
//    /// A statement's branch target by its index.
    /// A statement's branch target by its index.
//    Return(usize),
    Return(usize),
//}
}
//

//pub fn increment_builtin_counter<'ctx: 'a, 'a>(
pub fn increment_builtin_counter<'ctx: 'a, 'a>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    block: &'ctx Block<'ctx>,
    block: &'ctx Block<'ctx>,
//    location: Location<'ctx>,
    location: Location<'ctx>,
//    value: Value<'ctx, '_>,
    value: Value<'ctx, '_>,
//) -> crate::error::Result<Value<'ctx, 'a>> {
) -> crate::error::Result<Value<'ctx, 'a>> {
//    let k1 = block
    let k1 = block
//        .append_operation(arith::constant(
        .append_operation(arith::constant(
//            context,
            context,
//            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
//            location,
            location,
//        ))
        ))
//        .result(0)?
        .result(0)?
//        .into();
        .into();
//

//    Ok(block
    Ok(block
//        .append_operation(arith::addi(value, k1, location))
        .append_operation(arith::addi(value, k1, location))
//        .result(0)?
        .result(0)?
//        .into())
        .into())
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod tests {
mod tests {
//    use super::*;
    use super::*;
//    use crate::context::NativeContext;
    use crate::context::NativeContext;
//    use melior::ir::Type;
    use melior::ir::Type;
//

//    #[test]
    #[test]
//    fn switch_branch_arg_external_test() {
    fn switch_branch_arg_external_test() {
//        // Create a new context for MLIR operations
        // Create a new context for MLIR operations
//        let native_context = NativeContext::new();
        let native_context = NativeContext::new();
//        let context = native_context.context();
        let context = native_context.context();
//

//        // Create an unknown location in the context
        // Create an unknown location in the context
//        let location = Location::unknown(context);
        let location = Location::unknown(context);
//        // Create a new MLIR module with the unknown location
        // Create a new MLIR module with the unknown location
//        let module = Module::new(location);
        let module = Module::new(location);
//

//        // Create a new MLIR block and obtain its reference
        // Create a new MLIR block and obtain its reference
//        let region = Region::new();
        let region = Region::new();
//        let last_block = region.append_block(Block::new(&[]));
        let last_block = region.append_block(Block::new(&[]));
//

//        // Initialize the LibfuncHelper struct with various parameters
        // Initialize the LibfuncHelper struct with various parameters
//        let mut lib_func_helper = LibfuncHelper {
        let mut lib_func_helper = LibfuncHelper {
//            module: &module,
            module: &module,
//            init_block: &last_block,
            init_block: &last_block,
//            region: &region,
            region: &region,
//            blocks_arena: &Bump::new(),
            blocks_arena: &Bump::new(),
//            last_block: Cell::new(&last_block),
            last_block: Cell::new(&last_block),
//            branches: Vec::new(),
            branches: Vec::new(),
//            results: Vec::new(),
            results: Vec::new(),
//        };
        };
//

//        // Create an integer type with 32 bits
        // Create an integer type with 32 bits
//        let i32_type: Type = IntegerType::new(context, 32).into();
        let i32_type: Type = IntegerType::new(context, 32).into();
//        // Create a default block with the integer type and the unknown location
        // Create a default block with the integer type and the unknown location
//        let default_block = lib_func_helper.append_block(Block::new(&[(i32_type, location)]));
        let default_block = lib_func_helper.append_block(Block::new(&[(i32_type, location)]));
//

//        // Create a new MLIR block
        // Create a new MLIR block
//        let block = lib_func_helper.append_block(Block::new(&[]));
        let block = lib_func_helper.append_block(Block::new(&[]));
//

//        // Append a constant arithmetic operation to the block and obtain its result operand
        // Append a constant arithmetic operation to the block and obtain its result operand
//        let operand = block
        let operand = block
//            .append_operation(arith::constant(
            .append_operation(arith::constant(
//                context,
                context,
//                IntegerAttribute::new(i32_type, 1).into(),
                IntegerAttribute::new(i32_type, 1).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)
            .result(0)
//            .unwrap()
            .unwrap()
//            .into();
            .into();
//

//        // Loop to add branches and results to the LibfuncHelper struct
        // Loop to add branches and results to the LibfuncHelper struct
//        for _ in 0..20 {
        for _ in 0..20 {
//            // Push a default block and external operand to the branches vector
            // Push a default block and external operand to the branches vector
//            lib_func_helper
            lib_func_helper
//                .branches
                .branches
//                .push((default_block, vec![BranchArg::External(operand)]));
                .push((default_block, vec![BranchArg::External(operand)]));
//

//            // Push a new vector of result cells to the results vector
            // Push a new vector of result cells to the results vector
//            lib_func_helper.results.push([Cell::new(None)].into());
            lib_func_helper.results.push([Cell::new(None)].into());
//        }
        }
//

//        // Call the `switch` method of the LibfuncHelper struct and obtain the result
        // Call the `switch` method of the LibfuncHelper struct and obtain the result
//        let cf_switch = block.append_operation(
        let cf_switch = block.append_operation(
//            lib_func_helper
            lib_func_helper
//                .switch(
                .switch(
//                    context,
                    context,
//                    operand,
                    operand,
//                    (BranchTarget::Return(10), &[]),
                    (BranchTarget::Return(10), &[]),
//                    &[
                    &[
//                        (0, BranchTarget::Return(10), &[]),
                        (0, BranchTarget::Return(10), &[]),
//                        (1, BranchTarget::Return(10), &[]),
                        (1, BranchTarget::Return(10), &[]),
//                    ],
                    ],
//                    location,
                    location,
//                )
                )
//                .unwrap(),
                .unwrap(),
//        );
        );
//

//        // Assert that the switch operation is valid
        // Assert that the switch operation is valid
//        assert!(cf_switch.verify());
        assert!(cf_switch.verify());
//    }
    }
//

//    #[test]
    #[test]
//    fn switch_branch_arg_returned_test() {
    fn switch_branch_arg_returned_test() {
//        // Create a new context for MLIR operations
        // Create a new context for MLIR operations
//        let native_context = NativeContext::new();
        let native_context = NativeContext::new();
//        let context = native_context.context();
        let context = native_context.context();
//

//        // Create an unknown location in the context
        // Create an unknown location in the context
//        let location = Location::unknown(context);
        let location = Location::unknown(context);
//        // Create a new MLIR module with the unknown location
        // Create a new MLIR module with the unknown location
//        let module = Module::new(location);
        let module = Module::new(location);
//

//        // Create a new MLIR block and obtain its reference
        // Create a new MLIR block and obtain its reference
//        let region = Region::new();
        let region = Region::new();
//        let last_block = region.append_block(Block::new(&[]));
        let last_block = region.append_block(Block::new(&[]));
//

//        // Initialize the LibfuncHelper struct with various parameters
        // Initialize the LibfuncHelper struct with various parameters
//        let mut lib_func_helper = LibfuncHelper {
        let mut lib_func_helper = LibfuncHelper {
//            module: &module,
            module: &module,
//            init_block: &last_block,
            init_block: &last_block,
//            region: &region,
            region: &region,
//            blocks_arena: &Bump::new(),
            blocks_arena: &Bump::new(),
//            last_block: Cell::new(&last_block),
            last_block: Cell::new(&last_block),
//            branches: Vec::new(),
            branches: Vec::new(),
//            results: Vec::new(),
            results: Vec::new(),
//        };
        };
//

//        // Create an integer type with 32 bits
        // Create an integer type with 32 bits
//        let i32_type: Type = IntegerType::new(context, 32).into();
        let i32_type: Type = IntegerType::new(context, 32).into();
//        // Create a default block with the integer type and the unknown location
        // Create a default block with the integer type and the unknown location
//        let default_block = lib_func_helper.append_block(Block::new(&[(i32_type, location)]));
        let default_block = lib_func_helper.append_block(Block::new(&[(i32_type, location)]));
//

//        // Create a new MLIR block
        // Create a new MLIR block
//        let block = lib_func_helper.append_block(Block::new(&[]));
        let block = lib_func_helper.append_block(Block::new(&[]));
//

//        // Append a constant arithmetic operation to the block and obtain its result operand
        // Append a constant arithmetic operation to the block and obtain its result operand
//        let operand = block
        let operand = block
//            .append_operation(arith::constant(
            .append_operation(arith::constant(
//                context,
                context,
//                IntegerAttribute::new(i32_type, 1).into(),
                IntegerAttribute::new(i32_type, 1).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)
            .result(0)
//            .unwrap()
            .unwrap()
//            .into();
            .into();
//

//        // Loop to add branches and results to the LibfuncHelper struct
        // Loop to add branches and results to the LibfuncHelper struct
//        for _ in 0..20 {
        for _ in 0..20 {
//            // Push a default block and a returned operand index to the branches vector
            // Push a default block and a returned operand index to the branches vector
//            lib_func_helper
            lib_func_helper
//                .branches
                .branches
//                .push((default_block, vec![BranchArg::Returned(3)]));
                .push((default_block, vec![BranchArg::Returned(3)]));
//

//            // Push a new vector of result cells to the results vector
            // Push a new vector of result cells to the results vector
//            lib_func_helper.results.push([Cell::new(None)].into());
            lib_func_helper.results.push([Cell::new(None)].into());
//        }
        }
//

//        // Call the `switch` method of the LibfuncHelper struct and obtain the result
        // Call the `switch` method of the LibfuncHelper struct and obtain the result
//        let cf_switch = block.append_operation(
        let cf_switch = block.append_operation(
//            lib_func_helper
            lib_func_helper
//                .switch(
                .switch(
//                    context,
                    context,
//                    operand,
                    operand,
//                    (
                    (
//                        BranchTarget::Return(10),
                        BranchTarget::Return(10),
//                        &[operand, operand, operand, operand],
                        &[operand, operand, operand, operand],
//                    ),
                    ),
//                    &[
                    &[
//                        (0, BranchTarget::Return(10), &[]),
                        (0, BranchTarget::Return(10), &[]),
//                        (1, BranchTarget::Return(10), &[]),
                        (1, BranchTarget::Return(10), &[]),
//                    ],
                    ],
//                    location,
                    location,
//                )
                )
//                .unwrap(),
                .unwrap(),
//        );
        );
//

//        // Assert that the switch operation is valid
        // Assert that the switch operation is valid
//        assert!(cf_switch.verify());
        assert!(cf_switch.verify());
//

//        // Assert that the result in the LibfuncHelper at index 10 contains the expected operand
        // Assert that the result in the LibfuncHelper at index 10 contains the expected operand
//        assert_eq!(lib_func_helper.results[10][0], Cell::new(Some(operand)));
        assert_eq!(lib_func_helper.results[10][0], Cell::new(Some(operand)));
//

//        // Assert that the length of the results vector at index 10 is 1
        // Assert that the length of the results vector at index 10 is 1
//        assert_eq!(lib_func_helper.results[10].len(), 1);
        assert_eq!(lib_func_helper.results[10].len(), 1);
//    }
    }
//

//    #[test]
    #[test]
//    fn switch_branch_target_jump_test() {
    fn switch_branch_target_jump_test() {
//        // Create a new context for MLIR operations
        // Create a new context for MLIR operations
//        let native_context = NativeContext::new();
        let native_context = NativeContext::new();
//        let context = native_context.context();
        let context = native_context.context();
//

//        // Create an unknown location in the context
        // Create an unknown location in the context
//        let location = Location::unknown(context);
        let location = Location::unknown(context);
//        // Create a new MLIR module with the unknown location
        // Create a new MLIR module with the unknown location
//        let module = Module::new(location);
        let module = Module::new(location);
//

//        // Create a new MLIR block and obtain its reference
        // Create a new MLIR block and obtain its reference
//        let region = Region::new();
        let region = Region::new();
//        let last_block = region.append_block(Block::new(&[]));
        let last_block = region.append_block(Block::new(&[]));
//

//        // Initialize the LibfuncHelper struct with various parameters
        // Initialize the LibfuncHelper struct with various parameters
//        let mut lib_func_helper = LibfuncHelper {
        let mut lib_func_helper = LibfuncHelper {
//            module: &module,
            module: &module,
//            init_block: &last_block,
            init_block: &last_block,
//            region: &region,
            region: &region,
//            blocks_arena: &Bump::new(),
            blocks_arena: &Bump::new(),
//            last_block: Cell::new(&last_block),
            last_block: Cell::new(&last_block),
//            branches: Vec::new(),
            branches: Vec::new(),
//            results: Vec::new(),
            results: Vec::new(),
//        };
        };
//

//        // Create an integer type with 32 bits
        // Create an integer type with 32 bits
//        let i32_type: Type = IntegerType::new(context, 32).into();
        let i32_type: Type = IntegerType::new(context, 32).into();
//        // Create a default block with the integer type and the unknown location
        // Create a default block with the integer type and the unknown location
//        let default_block = lib_func_helper.append_block(Block::new(&[(i32_type, location)]));
        let default_block = lib_func_helper.append_block(Block::new(&[(i32_type, location)]));
//

//        // Create a new MLIR block
        // Create a new MLIR block
//        let block = lib_func_helper.append_block(Block::new(&[]));
        let block = lib_func_helper.append_block(Block::new(&[]));
//

//        // Append a constant arithmetic operation to the block and obtain its result operand
        // Append a constant arithmetic operation to the block and obtain its result operand
//        let operand = block
        let operand = block
//            .append_operation(arith::constant(
            .append_operation(arith::constant(
//                context,
                context,
//                IntegerAttribute::new(i32_type, 1).into(),
                IntegerAttribute::new(i32_type, 1).into(),
//                location,
                location,
//            ))
            ))
//            .result(0)
            .result(0)
//            .unwrap()
            .unwrap()
//            .into();
            .into();
//

//        // Loop to add branches and results to the LibfuncHelper struct
        // Loop to add branches and results to the LibfuncHelper struct
//        for _ in 0..20 {
        for _ in 0..20 {
//            // Push a default block and an empty vector of operands to the branches vector
            // Push a default block and an empty vector of operands to the branches vector
//            lib_func_helper.branches.push((default_block, Vec::new()));
            lib_func_helper.branches.push((default_block, Vec::new()));
//

//            // Push a new vector of result cells to the results vector
            // Push a new vector of result cells to the results vector
//            lib_func_helper.results.push([Cell::new(None)].into());
            lib_func_helper.results.push([Cell::new(None)].into());
//        }
        }
//

//        // Call the `switch` method of the LibfuncHelper struct and obtain the result
        // Call the `switch` method of the LibfuncHelper struct and obtain the result
//        let cf_switch = block.append_operation(
        let cf_switch = block.append_operation(
//            lib_func_helper
            lib_func_helper
//                .switch(
                .switch(
//                    context,
                    context,
//                    operand,
                    operand,
//                    (BranchTarget::Jump(default_block), &[operand]),
                    (BranchTarget::Jump(default_block), &[operand]),
//                    &[
                    &[
//                        (0, BranchTarget::Jump(default_block), &[operand]),
                        (0, BranchTarget::Jump(default_block), &[operand]),
//                        (1, BranchTarget::Jump(default_block), &[operand]),
                        (1, BranchTarget::Jump(default_block), &[operand]),
//                    ],
                    ],
//                    location,
                    location,
//                )
                )
//                .unwrap(),
                .unwrap(),
//        );
        );
//

//        // Assert that the switch operation is valid
        // Assert that the switch operation is valid
//        assert!(cf_switch.verify());
        assert!(cf_switch.verify());
//    }
    }
//}
}
