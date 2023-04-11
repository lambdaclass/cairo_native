use crate::libfuncs::lib_func_def::PositionalArg;

#[derive(Debug, Clone)]
pub struct UserFuncDef<'ctx> {
    pub(crate) args: Vec<PositionalArg<'ctx>>,
    pub(crate) return_types: Vec<PositionalArg<'ctx>>,
}
