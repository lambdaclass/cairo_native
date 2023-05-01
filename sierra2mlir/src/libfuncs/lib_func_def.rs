use std::collections::BTreeMap;

use itertools::Itertools;
use melior_next::ir::Type;

use crate::sierra_type::SierraType;

// MLIR implementations do not always use the same number of parameters
// As such, each LibFuncArg tracks both the type, and which parameter of the sierra libfunc it corresponds to for dataflow tracking
#[derive(Debug, Clone)]
pub struct PositionalArg<'ctx> {
    // 0-indexed location of the associated argument of the sierra libfunc
    pub(crate) loc: usize,
    pub(crate) ty: SierraType<'ctx>,
}

// The first step in implementating branching libfuncs is to select which branch to pick
// For some functions, such as X_is_zero, an input suffices, since the last branch is the switch's default block
// For others, a function must be called. E.g. for enum_match, a function needs to be called to get the enum's tag
// The ability to select which return from the selector function is the value to use allows for optimisation where
// selector and branch data can be calculated at the same time
#[derive(Debug, Clone)]
pub enum BranchSelector<'ctx> {
    Arg(PositionalArg<'ctx>),
    Call {
        name: String,
        args: Vec<PositionalArg<'ctx>>,
        return_type: Type<'ctx>,
        return_pos: usize,
    },
}

// Each branch needs to take some subset of the libfunc's arguments and turn them into the data passed to that branch
// There are 3 options:
//  Some branches receive no data, so there is nothing to do. This is represented as with option 2 but with an empty list
//  Some branches receive directly forwarded arguments of the libfuncs, such as with the X_is_zero functions
//  Some branches receive data that has been transformed from the arguments of the libfuncs
//    In a subset of these cases, the transformation occurs as part of the selector function,
//    in which case SelectorResult can be used to indicate which results to use and their type
#[derive(Debug, Clone)]
pub enum BranchProcessing<'ctx> {
    // The usize is for the index of the return value the arg is used for, allowing skipping of builtins
    Args (Vec<(PositionalArg<'ctx>, usize)>),
    Call { name: String, args: Vec<PositionalArg<'ctx>>, return_types: Vec<SierraType<'ctx>> },
    SelectorResult(Vec<(PositionalArg<'ctx>, usize)>),
}

impl<'ctx> BranchProcessing<'ctx> {
    pub fn none() -> Self {
        Self::Args(vec![])
    }
}

#[derive(Debug, Clone)]
pub enum SierraLibFunc<'ctx> {
    Branching { selector: BranchSelector<'ctx>, branch_processing: Vec<BranchProcessing<'ctx>> },
    Constant { ty: SierraType<'ctx>, value: String },
    Function { args: Vec<PositionalArg<'ctx>>, return_types: Vec<PositionalArg<'ctx>> },
    // Cases such as store_temp and dup that can be implemented purely at a dataflow level with no processing
    InlineDataflow(Vec<PositionalArg<'ctx>>),
}

impl<'ctx> SierraLibFunc<'ctx> {
    pub const fn create_constant(ty: SierraType<'ctx>, value: String) -> SierraLibFunc<'ctx> {
        Self::Constant { ty, value }
    }

    pub fn create_function_all_args(
        args: Vec<SierraType<'ctx>>,
        return_types: Vec<SierraType<'ctx>>,
    ) -> SierraLibFunc<'ctx> {
        Self::Function {
            args: args
                .iter()
                .enumerate()
                .map(|(loc, ty)| PositionalArg { loc, ty: ty.clone() })
                .collect_vec(),
            return_types: return_types
                .iter()
                .enumerate()
                .map(|(loc, ty)| PositionalArg { loc, ty: ty.clone() })
                .collect_vec(),
        }
    }

    pub fn get_args(&self) -> Vec<PositionalArg<'ctx>> {
        match self {
            SierraLibFunc::Function { args, return_types: _ } => args.clone(),
            SierraLibFunc::Constant { ty: _, value: _ } => vec![],
            SierraLibFunc::InlineDataflow(args) => args.clone(),
            SierraLibFunc::Branching { selector, branch_processing } => {
                // We don't want to return duplicates, just one PositionalArg for each argument of the libfunc that we need
                // As such, we're going to construct a map from the args used for the selector and the branch processing
                let mut args_used = BTreeMap::new();
                //
                match selector {
                    BranchSelector::Arg(arg) => {
                        args_used.insert(arg.loc, arg.clone());
                    }
                    BranchSelector::Call { args, .. } => {
                        args_used.extend(args.iter().map(|arg| (arg.loc, arg.clone())))
                    }
                }

                for processing in branch_processing.iter() {
                    match processing {
                        BranchProcessing::Args(args) => {
                            args_used.extend(args.iter().map(|(arg, _)| (arg.loc, arg.clone())));
                        }
                        BranchProcessing::Call { args, .. } => {
                            args_used.extend(args.iter().map(|arg| (arg.loc, arg.clone())));
                        },
                        BranchProcessing::SelectorResult(_) => {}
                    }
                }

                args_used.into_values().collect()
            }
        }
    }

    /// If true, the libfunc can be simply ignored during processing
    pub fn naively_skippable(&self) -> bool {
        match self {
            SierraLibFunc::Function { args: _, return_types } => return_types.is_empty(),
            SierraLibFunc::Constant { ty: _, value: _ } => false,
            SierraLibFunc::InlineDataflow(returns) => returns.is_empty(),
            SierraLibFunc::Branching { .. } => false,
        }
    }
}
