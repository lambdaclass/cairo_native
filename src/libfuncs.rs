//! # Compiler libfunc infrastructure
//!
//! Contains libfunc generation stuff (aka. the actual instructions).

use crate::{
    error::{CoreLibfuncBuilderError, CoreTypeBuilderError},
    metadata::MetadataStorage,
    types::TypeBuilder,
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{core::CoreConcreteLibfunc, GenericLibfunc, GenericType},
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::cf,
    ir::{Block, BlockRef, Location, Module, Operation, Region, Value, ValueLike},
    Context,
};
use std::{borrow::Cow, cell::Cell, error::Error, ops::Deref};

pub mod ap_tracking;
pub mod array;
pub mod bitwise;
pub mod r#bool;
pub mod r#box;
pub mod branch_align;
pub mod cast;
pub mod debug;
pub mod drop;
pub mod dup;
pub mod ec;
pub mod r#enum;
pub mod felt252;
pub mod felt252_dict;
pub mod felt252_dict_entry;
pub mod function_call;
pub mod gas;
pub mod mem;
pub mod nullable;
pub mod pedersen;
pub mod poseidon;
pub mod snapshot_take;
pub mod stark_net;
pub mod r#struct;
pub mod uint128;
pub mod uint16;
pub mod uint256;
pub mod uint32;
pub mod uint512;
pub mod uint64;
pub mod uint8;
pub mod unconditional_jump;
pub mod unwrap_non_zero;

/// Generation of MLIR operations from their Sierra counterparts.
///
/// All possible Sierra libfuncs must implement it. It is already implemented for all the core
/// libfuncs, contained in [CoreConcreteLibfunc].
pub trait LibfuncBuilder<TType, TLibfunc>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc<Concrete = Self>,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc>,
{
    /// Error type returned by this trait's methods.
    type Error: Error;

    /// Generate the MLIR operations.
    fn build<'ctx, 'this>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<TType, TLibfunc>,
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

impl<TType, TLibfunc> LibfuncBuilder<TType, TLibfunc> for CoreConcreteLibfunc
where
    TType: GenericType,
    TLibfunc: GenericLibfunc<Concrete = Self>,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
{
    type Error = CoreLibfuncBuilderError;

    fn build<'ctx, 'this>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<TType, TLibfunc>,
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
            Self::Bitwise(info) => {
                self::bitwise::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::BranchAlign(info) => self::branch_align::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Bool(selector) => self::r#bool::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Box(selector) => self::r#box::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Cast(selector) => self::cast::build(
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
            Self::Felt252(selector) => self::felt252::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::FunctionCall(info) => self::function_call::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Gas(selector) => self::gas::build(
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
            Self::Uint256(_) => todo!(),
            Self::Uint512(_) => todo!(),
            Self::Mem(selector) => self::mem::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Nullable(selector) => self::nullable::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::UnwrapNonZero(info) => self::unwrap_non_zero::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::UnconditionalJump(info) => self::unconditional_jump::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Enum(selector) => self::r#enum::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Struct(selector) => self::r#struct::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Felt252Dict(selector) => self::felt252_dict::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Felt252DictEntry(selector) => self::felt252_dict_entry::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Pedersen(selector) => self::pedersen::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Poseidon(_) => todo!(),
            Self::StarkNet(selector) => self::stark_net::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Debug(selector) => self::debug::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::SnapshotTake(info) => self::snapshot_take::build(
                context, registry, entry, location, helper, metadata, info,
            ),
        }
    }

    fn is_function_call(&self) -> Option<&FunctionId> {
        match self {
            CoreConcreteLibfunc::FunctionCall(info) => Some(&info.function.id),
            _ => None,
        }
    }
}

/// Helper struct which contains logic generation for extra MLIR blocks and branch operations to the
/// next statements.
///
/// Each branch index should be present in exactly one call a branching method (either
/// [`br`](#method.br), [`cond_br`](#method.cond_br) or [`switch`](#method.switch)).
///
/// This helper is necessary because the statement following the current one may not have the same
/// arguments as the results returned by the current statement. Because of that, a direct jump
/// cannot be made and some processing is required.
pub struct LibfuncHelper<'ctx, 'this>
where
    'this: 'ctx,
{
    pub(crate) module: &'this Module<'ctx>,
    pub(crate) init_block: &'this BlockRef<'ctx, 'this>,

    pub(crate) region: &'this Region<'ctx>,
    pub(crate) blocks_arena: &'this Bump,
    pub(crate) last_block: Cell<&'this BlockRef<'ctx, 'this>>,

    pub(crate) branches: Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>,
    pub(crate) results: Vec<Vec<Cell<Option<Value<'ctx, 'this>>>>>,
}

impl<'ctx, 'this> LibfuncHelper<'ctx, 'this>
where
    'this: 'ctx,
{
    pub(crate) fn results(self) -> impl Iterator<Item = Vec<Value<'ctx, 'this>>> {
        self.results
            .into_iter()
            .map(|x| x.into_iter().map(|x| x.into_inner().unwrap()).collect())
    }

    /// Return the initialization block.
    ///
    /// The init block is used for `llvm.alloca` instructions. It is guaranteed to not be executed
    /// multiple times on tail-recursive functions. This property allows generating tail-recursive
    /// functions that do not grow the stack.
    pub fn init_block(&self) -> &Block<'ctx> {
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
    pub fn br(
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
    pub fn cond_br(
        &self,
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
            unsafe { location.context().to_ref() },
            condition,
            block_true,
            block_false,
            &args_true,
            &args_false,
            location,
        )
    }

    /// Creates a conditional multi-branching operation, potentially jumping out of the libfunc and
    /// into the next statement.
    ///
    /// While generating a `switch` that doesn't jump out of the libfunc is possible, it should be
    /// avoided whenever possible. In those cases just use [melior::dialect::cf::switch].
    ///
    /// This method will also store the returned values so that they can be moved into the state and
    /// used later on when required.
    pub fn switch(
        &self,
        flag: Value<'ctx, 'this>,
        default: (BranchTarget<'ctx, '_>, &[Value<'ctx, 'this>]),
        branches: &[(i64, BranchTarget<'ctx, '_>, &[Value<'ctx, 'this>])],
        location: Location<'ctx>,
    ) -> Operation<'ctx> {
        let default_destination = match default.0 {
            BranchTarget::Jump(x) => (x, Cow::Borrowed(default.1)),
            BranchTarget::Return(i) => {
                let (successor, operands) = &self.branches[i];

                for (dst, src) in self.results[i].iter().zip(default.1) {
                    dst.replace(Some(*src));
                }

                let destination_operands = operands
                    .iter()
                    .copied()
                    .map(|op| match op {
                        BranchArg::External(x) => x,
                        BranchArg::Returned(i) => default.1[i],
                    })
                    .collect::<Vec<_>>();

                (*successor, Cow::Owned(destination_operands))
            }
        };

        let mut case_values = Vec::with_capacity(branches.len());
        let mut case_destinations = Vec::with_capacity(branches.len());
        for (flag, successor, operands) in branches {
            case_values.push(*flag);

            case_destinations.push(match *successor {
                BranchTarget::Jump(x) => (x, Cow::Borrowed(*operands)),
                BranchTarget::Return(i) => {
                    let (successor, operands) = &self.branches[i];

                    for (dst, src) in self.results[i].iter().zip(default.1) {
                        dst.replace(Some(*src));
                    }

                    let destination_operands = operands
                        .iter()
                        .copied()
                        .map(|op| match op {
                            BranchArg::External(x) => x,
                            BranchArg::Returned(i) => default.1[i],
                        })
                        .collect::<Vec<_>>();

                    (*successor, Cow::Owned(destination_operands))
                }
            });
        }

        cf::switch(
            unsafe { location.context().to_ref() },
            &case_values,
            flag,
            flag.r#type(),
            (default_destination.0, &default_destination.1),
            &case_destinations
                .iter()
                .map(|(x, y)| (*x, y.as_ref()))
                .collect::<Vec<_>>(),
            location,
        )
        .unwrap()
    }
}

impl<'ctx, 'this> Deref for LibfuncHelper<'ctx, 'this> {
    type Target = Module<'ctx>;

    fn deref(&self) -> &Self::Target {
        self.module
    }
}

#[derive(Clone, Copy)]
pub(crate) enum BranchArg<'ctx, 'this> {
    External(Value<'ctx, 'this>),
    Returned(usize),
}

/// A libfunc branching target.
///
/// May point to either a block within the same libfunc using [BranchTarget::Jump] or to one of the
/// statement's branches using [BranchTarget::Return] with the branch index.
#[derive(Clone, Copy)]
pub enum BranchTarget<'ctx, 'a> {
    /// A block within the current libfunc.
    Jump(&'a Block<'ctx>),
    /// A statement's branch target by its index.
    Return(usize),
}
