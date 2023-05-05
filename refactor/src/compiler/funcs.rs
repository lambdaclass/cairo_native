use super::{types::SierraTypeId, Compiler};
use cairo_lang_sierra::{
    ids::{FunctionId, VarId},
    program::{Function, StatementIdx},
};
use std::collections::BTreeMap;

pub type FuncStorage = BTreeMap<SierraFuncId, CompiledFunc>;

/// Compiled sierra function identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SierraFuncId(u64);

impl SierraFuncId {
    pub(super) const fn new(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug)]
pub struct CompiledFunc {
    pub(crate) _id: FunctionId,

    pub(crate) arguments: Vec<SierraTypeId>,
    pub(crate) return_types: Vec<SierraTypeId>,

    pub(crate) entry_point: StatementIdx,
    pub(crate) arg_mappings: Vec<VarId>,
}

impl CompiledFunc {
    pub fn new(compiler: &Compiler, func_declaration: &Function) -> Self {
        Self {
            _id: func_declaration.id.clone(),
            arguments: func_declaration
                .signature
                .param_types
                .iter()
                .map(|arg_id| {
                    let arg_id = SierraTypeId::new(arg_id.id);
                    *compiler
                        .compiled_types
                        .iter()
                        .find(|x| x.0 == &arg_id)
                        .unwrap()
                        .0
                })
                .collect(),
            return_types: func_declaration
                .signature
                .ret_types
                .iter()
                .map(|arg_id| {
                    let arg_id = SierraTypeId::new(arg_id.id);
                    *compiler
                        .compiled_types
                        .iter()
                        .find(|x| x.0 == &arg_id)
                        .unwrap()
                        .0
                })
                .collect(),
            entry_point: func_declaration.entry_point,
            arg_mappings: func_declaration
                .params
                .iter()
                .map(|x| x.id.clone())
                .collect(),
        }
    }
}
