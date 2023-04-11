use itertools::Itertools;

use crate::compiler::SierraType;

// MLIR implementations do not always use the same number of parameters
// As such, each LibFuncArg tracks both the type, and which parameter of the sierra libfunc it corresponds to for dataflow tracking
#[derive(Debug, Clone)]
pub struct LibFuncArg<'ctx> {
    // 0-indexed location of the associated argument of the sierra libfunc
    pub(crate) loc: usize,
    pub(crate) ty: SierraType<'ctx>,
}

#[derive(Debug, Clone)]
pub struct LibFuncDef<'ctx> {
    pub(crate) args: Vec<LibFuncArg<'ctx>>,
    // Sierra Libfuncs can 'return' different information depending on the flow target
    pub(crate) return_types: Vec<Vec<SierraType<'ctx>>>,
}

#[derive(Debug, Clone)]
pub struct ConstantLibFunc<'ctx> {
    pub(crate) ty: SierraType<'ctx>,
    pub(crate) value: String,
}

#[derive(Debug, Clone)]
pub enum SierraLibFunc<'ctx> {
    Function(LibFuncDef<'ctx>),
    Constant(ConstantLibFunc<'ctx>),
    // Cases such as store_temp and dup that can be implemented purely at a dataflow level with no processing
    InlineDataflow(Vec<LibFuncArg<'ctx>>),
}

impl<'ctx> SierraLibFunc<'ctx> {
    pub const fn create_constant(ty: SierraType<'ctx>, value: String) -> SierraLibFunc<'ctx> {
        Self::Constant(ConstantLibFunc { ty, value })
    }

    pub fn create_simple(
        args: Vec<SierraType<'ctx>>,
        return_types: Vec<SierraType<'ctx>>,
    ) -> SierraLibFunc<'ctx> {
        Self::Function(LibFuncDef {
            args: args
                .iter()
                .enumerate()
                .map(|(loc, ty)| LibFuncArg { loc, ty: ty.clone() })
                .collect_vec(),
            return_types: vec![return_types],
        })
    }

    pub fn get_args(&self) -> Vec<LibFuncArg<'ctx>> {
        match self {
            SierraLibFunc::Function(LibFuncDef { args, return_types: _ }) => {
                args.iter().cloned().collect_vec()
            }
            SierraLibFunc::Constant(_) => vec![],
            SierraLibFunc::InlineDataflow(args) => args.clone(),
        }
    }

    /// If true, the libfunc can be simply ignored during processing
    pub fn naively_skippable(&self) -> bool {
        match self {
            SierraLibFunc::Function(LibFuncDef { args: _, return_types }) => {
                return_types.iter().all(|x| x.is_empty())
            }
            SierraLibFunc::Constant(_) => false,
            SierraLibFunc::InlineDataflow(args) => args.is_empty(),
        }
    }

    pub fn as_lib_func_def(&self) -> LibFuncDef {
        match self {
            SierraLibFunc::Function(lib_func_def) => lib_func_def.clone(),
            SierraLibFunc::Constant(_) => {
                panic!("Expected SierraLibFunc to be function, got constant")
            }
            SierraLibFunc::InlineDataflow(_) => {
                panic!("Expected SierraLibFunc to be function, got inline dataflow")
            }
        }
    }
}
