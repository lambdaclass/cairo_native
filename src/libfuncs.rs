use crate::types::TypeBuilder;
use cairo_lang_sierra::{
    extensions::{core::CoreConcreteLibfunc, GenericLibfunc, GenericType},
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::cf,
    ir::{Block, Location, Module, Operation, Value},
    Context,
};
use std::{cell::Cell, error::Error};

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

// pub struct LibfuncBuilderContext<'ctx, 'this, TType, TLibfunc>
// where
//     TType: GenericType,
//     TLibfunc: GenericLibfunc,
//     <TType as GenericType>::Concrete: TypeBuilder,
//     <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
// {
//     context: &'ctx Context,
//     registry: &'this ProgramRegistry<TType, TLibfunc>,

//     module: &'this Module<'ctx>,
//     entry: &'this Block<'ctx>,
//     location: Location<'ctx>,
//     branches: Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>,
//     results: Rc<Vec<Vec<Cell<Option<Value<'ctx, 'this>>>>>>,
// }

// impl<'ctx, 'this, TType, TLibfunc> LibfuncBuilderContext<'ctx, 'this, TType, TLibfunc>
// where
//     TType: GenericType,
//     TLibfunc: GenericLibfunc,
//     <TType as GenericType>::Concrete: TypeBuilder,
//     <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
// {
//     pub fn new(
//         context: &'ctx Context,
//         registry: &'this ProgramRegistry<TType, TLibfunc>,
//         module: &'this Module<'ctx>,
//         entry: &'this Block<'ctx>,
//         location: Location<'ctx>,
//         branches: Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>,
//         results: Rc<Vec<Vec<Cell<Option<Value<'ctx, 'this>>>>>>,
//     ) -> Self {
//         Self {
//             context,
//             registry,
//             module,
//             entry,
//             location,
//             branches,
//             results,
//         }
//     }

//     pub fn context(&self) -> &'ctx Context {
//         self.context
//     }

//     pub fn registry(&self) -> &'this ProgramRegistry<TType, TLibfunc> {
//         self.registry
//     }

//     pub fn module(&self) -> &'this Module<'ctx> {
//         self.module
//     }

//     pub fn entry(&self) -> &'this Block<'ctx> {
//         self.entry
//     }

//     pub fn location(&self) -> Location<'ctx> {
//         self.location
//     }

//     pub fn br(&self, branch: usize, results: &[Value<'ctx, 'this>]) -> Operation<'ctx> {
//         let (successor, operands) = &self.branches[branch];

//         for (dst, src) in self.results[branch].iter().zip(results) {
//             dst.replace(Some(*src));
//         }

//         let destination_operands = operands
//             .iter()
//             .copied()
//             .map(|op| match op {
//                 BranchArg::External(x) => x,
//                 BranchArg::Returned(i) => results[i],
//             })
//             .collect::<Vec<_>>();

//         cf::br(successor, &destination_operands, self.location())
//     }

//     // TODO: Allow one block to be libfunc-internal.
//     pub fn cond_br(
//         &self,
//         condition: Value<'ctx, 'this>,
//         branches: (usize, usize),
//         results: &[Value<'ctx, 'this>],
//     ) -> Operation<'ctx> {
//         let (block_true, args_true) = {
//             let (successor, operands) = &self.branches[branches.0];

//             for (dst, src) in self.results[branches.0].iter().zip(results) {
//                 dst.replace(Some(*src));
//             }

//             let destination_operands = operands
//                 .iter()
//                 .copied()
//                 .map(|op| match op {
//                     BranchArg::External(x) => x,
//                     BranchArg::Returned(i) => results[i],
//                 })
//                 .collect::<Vec<_>>();

//             (*successor, destination_operands)
//         };

//         let (block_false, args_false) = {
//             let (successor, operands) = &self.branches[branches.1];

//             for (dst, src) in self.results[branches.1].iter().zip(results) {
//                 dst.replace(Some(*src));
//             }

//             let destination_operands = operands
//                 .iter()
//                 .copied()
//                 .map(|op| match op {
//                     BranchArg::External(x) => x,
//                     BranchArg::Returned(i) => results[i],
//                 })
//                 .collect::<Vec<_>>();

//             (*successor, destination_operands)
//         };

//         cf::cond_br(
//             self.context(),
//             condition,
//             block_true,
//             block_false,
//             &args_true,
//             &args_false,
//             self.location(),
//         )
//     }
// }

// #[derive(Clone, Copy)]
// pub enum BranchArg<'ctx, 'this> {
//     External(Value<'ctx, 'this>),
//     Returned(usize),
// }

pub trait LibfuncBuilder {
    type Error: Error;

    fn build<'ctx, 'this, TType, TLibfunc>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<TType, TLibfunc>,
        entry: &'this Block<'ctx>,
        location: Location<'ctx>,
        helper: &LibfuncHelper<'ctx, 'this>,
    ) -> Result<(), Self::Error>
    where
        TType: GenericType,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder,
        <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder;
}

impl LibfuncBuilder for CoreConcreteLibfunc {
    type Error = std::convert::Infallible;

    fn build<'ctx, 'this, TType, TLibfunc>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<TType, TLibfunc>,
        entry: &'this Block<'ctx>,
        location: Location<'ctx>,
        helper: &LibfuncHelper<'ctx, 'this>,
    ) -> Result<(), Self::Error>
    where
        TType: GenericType,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder,
        <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
    {
        match self {
            Self::ApTracking(_) => todo!(),
            Self::Array(_) => todo!(),
            Self::Bitwise(_) => todo!(),
            Self::BranchAlign(info) => {
                self::branch_align::build(context, registry, entry, location, helper, info)
            }
            Self::Bool(_) => todo!(),
            Self::Box(_) => todo!(),
            Self::Cast(_) => todo!(),
            Self::Drop(info) => self::drop::build(context, registry, entry, location, helper, info),
            Self::Dup(_) => todo!(),
            Self::Ec(_) => todo!(),
            Self::Felt252(selector) => {
                self::felt252::build(context, registry, entry, location, helper, selector)
            }
            Self::FunctionCall(info) => {
                self::function_call::build(context, registry, entry, location, helper, info)
            }
            Self::Gas(_) => todo!(),
            Self::Uint8(selector) => {
                self::uint8::build(context, registry, entry, location, helper, selector)
            }
            Self::Uint16(_) => todo!(),
            Self::Uint32(selector) => {
                self::uint32::build(context, registry, entry, location, helper, selector)
            }
            Self::Uint64(_) => todo!(),
            Self::Uint128(_) => todo!(),
            Self::Uint256(_) => todo!(),
            Self::Uint512(_) => todo!(),
            Self::Mem(selector) => {
                self::mem::build(context, registry, entry, location, helper, selector)
            }
            Self::Nullable(_) => todo!(),
            Self::UnwrapNonZero(_) => todo!(),
            Self::UnconditionalJump(info) => {
                self::unconditional_jump::build(context, registry, entry, location, helper, info)
            }
            Self::Enum(_) => todo!(),
            Self::Struct(selector) => {
                self::r#struct::build(context, registry, entry, location, helper, selector)
            }
            Self::Felt252Dict(_) => todo!(),
            Self::Felt252DictEntry(_) => todo!(),
            Self::Pedersen(_) => todo!(),
            Self::Poseidon(_) => todo!(),
            Self::StarkNet(_) => todo!(),
            Self::Debug(_) => todo!(),
            Self::SnapshotTake(_) => todo!(),
        }
    }
}

pub struct LibfuncHelper<'ctx, 'this> {
    pub(crate) _module: &'this Module<'ctx>,
    pub(crate) branches: Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>,
    pub(crate) results: Vec<Vec<Cell<Option<Value<'ctx, 'this>>>>>,
}

impl<'ctx, 'this> LibfuncHelper<'ctx, 'this> {
    pub(crate) fn results(self) -> impl Iterator<Item = Vec<Value<'ctx, 'this>>> {
        self.results
            .into_iter()
            .map(|x| x.into_iter().map(|x| x.into_inner().unwrap()).collect())
    }

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

    // TODO: Allow one block to be libfunc-internal.
    pub fn cond_br(
        &self,
        condition: Value<'ctx, 'this>,
        branches: (usize, usize),
        results: &[Value<'ctx, 'this>],
        location: Location<'ctx>,
    ) -> Operation<'ctx> {
        let (block_true, args_true) = {
            let (successor, operands) = &self.branches[branches.0];

            for (dst, src) in self.results[branches.0].iter().zip(results) {
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

            (*successor, destination_operands)
        };

        let (block_false, args_false) = {
            let (successor, operands) = &self.branches[branches.1];

            for (dst, src) in self.results[branches.1].iter().zip(results) {
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
}

#[derive(Clone, Copy)]
pub enum BranchArg<'ctx, 'this> {
    External(Value<'ctx, 'this>),
    Returned(usize),
}
