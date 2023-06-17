use self::{
    funcs::find_func, libfunc_declarations::find_libfunc_declaration,
    statements::find_all_statements, type_declarations::find_type_declaration,
};
use cairo_lang_compiler::db::RootDatabase;
use cairo_lang_defs::diagnostic_utils::StableLocation;
use cairo_lang_diagnostics::DiagnosticAdded;
use cairo_lang_sierra::{
    ids::{ConcreteLibfuncId, ConcreteTypeId, FunctionId},
    program::{Program, StatementIdx},
};
use std::collections::HashMap;

mod funcs;
mod libfunc_declarations;
mod statements;
mod type_declarations;

#[derive(Clone, Debug)]
pub struct DebugInfo {
    pub type_declarations: HashMap<ConcreteTypeId, StableLocation>,
    pub libfunc_declarations: HashMap<ConcreteLibfuncId, StableLocation>,
    pub statements: HashMap<StatementIdx, StableLocation>,
    pub funcs: HashMap<FunctionId, StableLocation>,
}

impl DebugInfo {
    pub fn extract(db: &RootDatabase, program: &Program) -> Result<Self, DiagnosticAdded> {
        let type_declarations = program
            .type_declarations
            .iter()
            .filter_map(|type_declaration| {
                find_type_declaration(db, type_declaration)
                    .map(|x| x.map(|location| (type_declaration.id.clone(), location)))
                    .transpose()
            })
            .try_collect()?;

        let libfunc_declarations = program
            .libfunc_declarations
            .iter()
            .filter_map(|libfunc_declaration| {
                find_libfunc_declaration(db, libfunc_declaration)
                    .map(|x| x.map(|location| (libfunc_declaration.id.clone(), location)))
                    .transpose()
            })
            .try_collect::<HashMap<_, _>>()?;

        let statements =
            find_all_statements(db, |id| libfunc_declarations.contains_key(id), program)?;

        let funcs = program
            .funcs
            .iter()
            .map(|function| Ok((function.id.clone(), find_func(db, function)?)))
            .try_collect()?;

        Ok(Self {
            type_declarations,
            libfunc_declarations,
            statements,
            funcs,
        })
    }
}
