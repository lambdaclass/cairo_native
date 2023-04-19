use itertools::Itertools;

use crate::compiler::SierraType;

// MLIR implementations do not always use the same number of parameters
// As such, each LibFuncArg tracks both the type, and which parameter of the sierra libfunc it corresponds to for dataflow tracking
#[derive(Debug, Clone)]
pub struct PositionalArg<'ctx> {
    // 0-indexed location of the associated argument of the sierra libfunc
    pub(crate) loc: usize,
    pub(crate) ty: SierraType<'ctx>,
}

#[derive(Debug, Clone)]
pub enum SierraLibFunc<'ctx> {
    Branching { args: Vec<PositionalArg<'ctx>>, return_types: Vec<Vec<PositionalArg<'ctx>>> },
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
            SierraLibFunc::Branching { args, return_types: _ } => args.clone(),
        }
    }

    pub fn get_return_types(&self) -> Vec<Vec<PositionalArg<'ctx>>> {
        match self {
            SierraLibFunc::Function { return_types, .. } => vec![return_types.clone()],
            SierraLibFunc::Constant { .. } => vec![],
            SierraLibFunc::InlineDataflow(_) => vec![],
            SierraLibFunc::Branching { return_types, .. } => return_types.clone(),
        }
    }

    /// If true, the libfunc can be simply ignored during processing
    pub fn naively_skippable(&self) -> bool {
        match self {
            SierraLibFunc::Function { args: _, return_types } => return_types.is_empty(),
            SierraLibFunc::Constant { ty: _, value: _ } => false,
            SierraLibFunc::InlineDataflow(returns) => returns.is_empty(),
            SierraLibFunc::Branching { args: _, return_types } => {
                return_types.len() == 1 && return_types[0].is_empty()
            }
        }
    }
}
