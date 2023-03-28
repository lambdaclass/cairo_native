use itertools::Itertools;

use crate::compiler::SierraType;

// MLIR implementations do not always use the same number of parameters
// As such, each LibFuncArg tracks both the type, and which parameter of the sierra libfunc it corresponds to for dataflow tracking
#[derive(Debug, Clone)]
pub struct LibFuncArg<'ctx> {
    pub(crate) loc: usize,
    pub(crate) ty: SierraType<'ctx>
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
}

impl<'ctx> SierraLibFunc<'ctx> {
    pub fn create_constant(ty: SierraType<'ctx>, value: String) -> SierraLibFunc<'ctx> {
        Self::Constant(ConstantLibFunc{ty, value})
    }

    pub fn create_simple(args: Vec<SierraType<'ctx>>, return_types: Vec<SierraType<'ctx>>) -> SierraLibFunc<'ctx> {
        Self::Function(LibFuncDef{
            args: args.iter().enumerate().map(|(loc, ty)| LibFuncArg{loc, ty: ty.clone()}).collect_vec(),
            return_types: vec![return_types],
        })
    }

    pub fn get_args(&self) -> Vec<&LibFuncArg> {
        match self {
            SierraLibFunc::Function(LibFuncDef{ args, return_types: _ }) => args.iter().collect_vec(),
            SierraLibFunc::Constant(_) => vec![],
        }
    }
}
