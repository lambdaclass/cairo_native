//! # Compiler libfunc infrastructure
//!
//! Contains libfunc generation stuff (aka. the actual instructions).

use crate::{error::Error as CoreLibfuncBuilderError, metadata::MetadataStorage, utils::BlockExt};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::core::{CoreConcreteLibfunc, CoreLibfunc, CoreType},
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, cf},
    ir::{Block, BlockRef, Location, Module, Operation, Region, Value},
    Context,
};
use num_bigint::BigInt;
use std::{cell::Cell, error::Error, ops::Deref};

mod ap_tracking;
mod array;
mod bitwise;
mod r#bool;
mod bounded_int;
mod r#box;
mod branch_align;
mod bytes31;
mod cast;
mod circuit;
mod r#const;
mod coupon;
mod debug;
mod drop;
mod dup;
mod ec;
mod r#enum;
mod felt252;
mod felt252_dict;
mod felt252_dict_entry;
mod function_call;
mod gas;
mod mem;
mod nullable;
mod pedersen;
mod poseidon;
mod sint128;
mod sint16;
mod sint32;
mod sint64;
mod sint8;
mod snapshot_take;
mod starknet;
mod r#struct;
mod uint128;
mod uint16;
mod uint256;
mod uint32;
mod uint512;
mod uint64;
mod uint8;
mod unconditional_jump;
mod unwrap_non_zero;

/// Generation of MLIR operations from their Sierra counterparts.
///
/// All possible Sierra libfuncs must implement it. It is already implemented for all the core
/// libfuncs, contained in [CoreConcreteLibfunc].
pub trait LibfuncBuilder {
    /// Error type returned by this trait's methods.
    type Error: Error;

    /// Generate the MLIR operations.
    fn build<'ctx, 'this>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        entry: &'this Block<'ctx>,
        location: Location<'ctx>,
        helper: &LibfuncHelper<'ctx, 'this>,
        metadata: &mut MetadataStorage,
    ) -> Result<(), Self::Error>;

    /// Return the target function if the statement is a function call.
    ///
    /// This is used by the compiler to check whether a statement is a function call and apply the
    /// tail recursion logic.
    fn is_function_call(&self) -> Option<&FunctionId>;
}

impl LibfuncBuilder for CoreConcreteLibfunc {
    type Error = CoreLibfuncBuilderError;

    fn build<'ctx, 'this>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        entry: &'this Block<'ctx>,
        location: Location<'ctx>,
        helper: &LibfuncHelper<'ctx, 'this>,
        metadata: &mut MetadataStorage,
    ) -> Result<(), Self::Error> {
        match self {
            Self::ApTracking(selector) => self::ap_tracking::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Array(selector) => self::array::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::BranchAlign(info) => self::branch_align::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Bool(selector) => self::r#bool::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Box(selector) => self::r#box::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Bytes31(selector) => self::bytes31::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Cast(selector) => self::cast::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Const(selector) => self::r#const::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Debug(selector) => self::debug::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Drop(info) => {
                self::drop::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Dup(info) => {
                self::dup::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Ec(selector) => self::ec::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Enum(selector) => self::r#enum::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Felt252(selector) => self::felt252::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Felt252Dict(selector) => self::felt252_dict::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Felt252DictEntry(selector) => self::felt252_dict_entry::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::FunctionCall(info) => self::function_call::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Gas(selector) => self::gas::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Mem(selector) => self::mem::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Nullable(selector) => self::nullable::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Pedersen(selector) => self::pedersen::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Poseidon(selector) => self::poseidon::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Sint8(info) => {
                self::sint8::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Sint16(info) => {
                self::sint16::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Sint32(info) => {
                self::sint32::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Sint64(info) => {
                self::sint64::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Sint128(info) => {
                self::sint128::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::SnapshotTake(info) => self::snapshot_take::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::StarkNet(selector) => self::starknet::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Struct(selector) => self::r#struct::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint8(selector) => self::uint8::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint16(selector) => self::uint16::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint32(selector) => self::uint32::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint64(selector) => self::uint64::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint128(selector) => self::uint128::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint256(selector) => self::uint256::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint512(selector) => self::uint512::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::UnconditionalJump(info) => self::unconditional_jump::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::UnwrapNonZero(info) => self::unwrap_non_zero::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Coupon(info) => {
                self::coupon::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::CouponCall(info) => self::function_call::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Circuit(info) => {
                self::circuit::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::BoundedInt(info) => {
                self::bounded_int::build(context, registry, entry, location, helper, metadata, info)
            }
        }
    }

    fn is_function_call(&self) -> Option<&FunctionId> {
        match self {
            CoreConcreteLibfunc::FunctionCall(info) => Some(&info.function.id),
            CoreConcreteLibfunc::CouponCall(info) => Some(&info.function.id),
            _ => None,
        }
    }
}

/// Helper struct which contains logic generation for extra MLIR blocks and branch operations to the
/// next statements.
///
/// Each branch index should be present in exactly one call a branching method (either
/// [`br`](#method.br) or [`cond_br`](#method.cond_br)).
///
/// This helper is necessary because the statement following the current one may not have the same
/// arguments as the results returned by the current statement. Because of that, a direct jump
/// cannot be made and some processing is required.
pub(crate) struct LibfuncHelper<'ctx, 'this>
where
    'this: 'ctx,
{
    pub module: &'this Module<'ctx>,
    pub init_block: &'this BlockRef<'ctx, 'this>,

    pub region: &'this Region<'ctx>,
    pub blocks_arena: &'this Bump,
    pub last_block: Cell<&'this BlockRef<'ctx, 'this>>,

    pub branches: Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>,
    pub results: Vec<Vec<Cell<Option<Value<'ctx, 'this>>>>>,
}

impl<'ctx, 'this> LibfuncHelper<'ctx, 'this>
where
    'this: 'ctx,
{
    #[doc(hidden)]
    pub(crate) fn results(self) -> impl Iterator<Item = Vec<Value<'ctx, 'this>>> {
        self.results.into_iter().enumerate().map(|(branch_idx, x)| {
            x.into_iter()
                .enumerate()
                .map(|(arg_idx, x)| {
                    x.into_inner().unwrap_or_else(|| {
                        panic!("Argument #{arg_idx} of branch {branch_idx} doesn't have a value.")
                    })
                })
                .collect()
        })
    }

    /// Return the initialization block.
    ///
    /// The init block is used for `llvm.alloca` instructions. It is guaranteed to not be executed
    /// multiple times on tail-recursive functions. This property allows generating tail-recursive
    /// functions that do not grow the stack.
    pub fn init_block(&self) -> &'this Block<'ctx> {
        self.init_block
    }

    /// Inserts a new block after all the current libfunc's blocks.
    pub fn append_block(&self, block: Block<'ctx>) -> &'this Block<'ctx> {
        let block = self
            .region
            .insert_block_after(*self.last_block.get(), block);

        let block_ref: &'this mut BlockRef<'ctx, 'this> = self.blocks_arena.alloc(block);
        self.last_block.set(block_ref);

        block_ref
    }

    /// Creates an unconditional branching operation out of the libfunc and into the next statement.
    ///
    /// This method will also store the returned values so that they can be moved into the state and
    /// used later on when required.
    fn br(
        &self,
        branch: usize,
        results: &[Value<'ctx, 'this>],
        location: Location<'ctx>,
    ) -> Operation<'ctx> {
        let (successor, operands) = &self.branches[branch];

        for (dst, src) in self.results[branch].iter().zip(results) {
            dst.replace(Some(*src));
        }

        let destination_operands = operands
            .iter()
            .copied()
            .map(|op| match op {
                BranchArg::External(x) => x,
                BranchArg::Returned(i) => results[i],
            })
            .collect::<Vec<_>>();

        cf::br(successor, &destination_operands, location)
    }

    /// Creates a conditional binary branching operation, potentially jumping out of the libfunc and
    /// into the next statement.
    ///
    /// While generating a `cond_br` that doesn't jump out of the libfunc is possible, it should be
    /// avoided whenever possible. In those cases just use [melior::dialect::cf::cond_br].
    ///
    /// This method will also store the returned values so that they can be moved into the state and
    /// used later on when required.
    // TODO: Allow one block to be libfunc-internal.
    fn cond_br(
        &self,
        context: &'ctx Context,
        condition: Value<'ctx, 'this>,
        branches: [usize; 2],
        results: [&[Value<'ctx, 'this>]; 2],
        location: Location<'ctx>,
    ) -> Operation<'ctx> {
        let (block_true, args_true) = {
            let (successor, operands) = &self.branches[branches[0]];

            for (dst, src) in self.results[branches[0]].iter().zip(results[0]) {
                dst.replace(Some(*src));
            }

            let destination_operands = operands
                .iter()
                .copied()
                .map(|op| match op {
                    BranchArg::External(x) => x,
                    BranchArg::Returned(i) => results[0][i],
                })
                .collect::<Vec<_>>();

            (*successor, destination_operands)
        };

        let (block_false, args_false) = {
            let (successor, operands) = &self.branches[branches[1]];

            for (dst, src) in self.results[branches[1]].iter().zip(results[1]) {
                dst.replace(Some(*src));
            }

            let destination_operands = operands
                .iter()
                .copied()
                .map(|op| match op {
                    BranchArg::External(x) => x,
                    BranchArg::Returned(i) => results[1][i],
                })
                .collect::<Vec<_>>();

            (*successor, destination_operands)
        };

        cf::cond_br(
            context,
            condition,
            block_true,
            block_false,
            &args_true,
            &args_false,
            location,
        )
    }
}

impl<'ctx, 'this> Deref for LibfuncHelper<'ctx, 'this> {
    type Target = Module<'ctx>;

    fn deref(&self) -> &Self::Target {
        self.module
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BranchArg<'ctx, 'this> {
    External(Value<'ctx, 'this>),
    Returned(usize),
}

fn increment_builtin_counter<'ctx: 'a, 'a>(
    context: &'ctx Context,
    block: &'ctx Block<'ctx>,
    location: Location<'ctx>,
    value: Value<'ctx, '_>,
) -> crate::error::Result<Value<'ctx, 'a>> {
    increment_builtin_counter_by(context, block, location, value, 1)
}

fn increment_builtin_counter_by<'ctx: 'a, 'a>(
    context: &'ctx Context,
    block: &'ctx Block<'ctx>,
    location: Location<'ctx>,
    value: Value<'ctx, '_>,
    amount: impl Into<BigInt>,
) -> crate::error::Result<Value<'ctx, 'a>> {
    block.append_op_result(arith::addi(
        value,
        block.const_int(context, location, amount, 64)?,
        location,
    ))
}
