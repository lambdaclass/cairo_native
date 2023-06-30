use self::{
    funcs::find_func, libfunc_declarations::find_libfunc_declaration,
    statements::find_all_statements, type_declarations::find_type_declaration,
};
use cairo_lang_compiler::db::RootDatabase;
use cairo_lang_defs::diagnostic_utils::StableLocation;
use cairo_lang_diagnostics::DiagnosticAdded;
use cairo_lang_filesystem::{db::FilesGroup, ids::FileLongId};
use cairo_lang_sierra::{
    ids::{ConcreteLibfuncId, ConcreteTypeId, FunctionId},
    program::{Program, StatementIdx},
};
use melior::{ir::Location, Context};
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

#[derive(Clone, Debug)]
pub struct DebugLocations<'c> {
    pub type_declarations: HashMap<ConcreteTypeId, Location<'c>>,
    pub libfunc_declarations: HashMap<ConcreteLibfuncId, Location<'c>>,
    pub statements: HashMap<StatementIdx, Location<'c>>,
    pub funcs: HashMap<FunctionId, Location<'c>>,
}

impl<'c> DebugLocations<'c> {
    pub fn extract(context: &'c Context, db: &RootDatabase, debug_info: &DebugInfo) -> Self {
        let type_declarations = debug_info
            .type_declarations
            .iter()
            .map(|(type_id, stable_loc)| {
                (
                    type_id.clone(),
                    extract_location_from_stable_loc(context, db, *stable_loc),
                )
            })
            .collect();

        let libfunc_declarations = debug_info
            .libfunc_declarations
            .iter()
            .map(|(libfunc_id, stable_loc)| {
                (
                    libfunc_id.clone(),
                    extract_location_from_stable_loc(context, db, *stable_loc),
                )
            })
            .collect();

        let statements = debug_info
            .statements
            .iter()
            .map(|(statement_idx, stable_loc)| {
                (
                    *statement_idx,
                    extract_location_from_stable_loc(context, db, *stable_loc),
                )
            })
            .collect();

        let funcs = debug_info
            .funcs
            .iter()
            .map(|(function_id, stable_loc)| {
                (
                    function_id.clone(),
                    extract_location_from_stable_loc(context, db, *stable_loc),
                )
            })
            .collect();

        Self {
            type_declarations,
            libfunc_declarations,
            statements,
            funcs,
        }
    }
}

fn extract_location_from_stable_loc<'c>(
    context: &'c Context,
    db: &RootDatabase,
    stable_loc: StableLocation,
) -> Location<'c> {
    let diagnostic_location = stable_loc.diagnostic_location(db);

    let path = match db.lookup_intern_file(diagnostic_location.file_id) {
        FileLongId::OnDisk(path) => path,
        FileLongId::Virtual(_) => return Location::unknown(context),
    };

    let pos = diagnostic_location
        .span
        .start
        .position_in_file(db, diagnostic_location.file_id)
        .unwrap();

    Location::new(context, path.to_str().unwrap(), pos.line, pos.col)
}
