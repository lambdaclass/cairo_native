//! Extracts useful debugging information from cairo programs to make it available to the generated MLIR.

use self::{
    funcs::find_func, libfunc_declarations::find_libfunc_declaration,
    statements::find_all_statements, type_declarations::find_type_declaration,
};
use cairo_lang_compiler::db::RootDatabase;
use cairo_lang_defs::diagnostic_utils::StableLocation;
use cairo_lang_diagnostics::DiagnosticAdded;
use cairo_lang_filesystem::{db::FilesGroup, ids::FileLongId};
use cairo_lang_lowering::ids::LocationId;
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
    pub statements: HashMap<StatementIdx, LocationId>,
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
            .collect::<Result<HashMap<_, _>, DiagnosticAdded>>()?;

        let libfunc_declarations = program
            .libfunc_declarations
            .iter()
            .filter_map(|libfunc_declaration| {
                find_libfunc_declaration(db, libfunc_declaration)
                    .map(|x| x.map(|location| (libfunc_declaration.id.clone(), location)))
                    .transpose()
            })
            .collect::<Result<HashMap<_, _>, DiagnosticAdded>>()?;

        let statements =
            find_all_statements(db, |id| libfunc_declarations.contains_key(id), program)?;

        let funcs = program
            .funcs
            .iter()
            .map(|function| Ok((function.id.clone(), find_func(db, function)?)))
            .collect::<Result<HashMap<_, _>, DiagnosticAdded>>()?;

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
            .map(|(statement_idx, location_id)| {
                (
                    *statement_idx,
                    extract_location_from_stable_loc(context, db, location_id.all_locations(db)[0]),
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

    Location::new(context, &path.to_string_lossy(), pos.line, pos.col)
}

#[cfg(test)]
mod test {
    use std::fs;

    use super::*;
    use crate::context::NativeContext;
    use cairo_lang_compiler::{compile_prepared_db, project::setup_project, CompilerConfig};
    use rstest::*;

    #[fixture]
    fn program() -> (Program, RootDatabase) {
        // Define a dummy program for testing
        let program_str = stringify! {
            use core::num::traits::Sqrt;
            fn run_test() -> u128 {
                let a: u128 = 1;
                a.sqrt().into()
            }
        };

        let mut program_file = tempfile::Builder::new()
            .prefix("test_")
            .suffix(".cairo")
            .tempfile()
            .unwrap();
        fs::write(&mut program_file, program_str).unwrap();

        let mut db = RootDatabase::builder().detect_corelib().build().unwrap();
        let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
        let sirrra_program = compile_prepared_db(
            &mut db,
            main_crate_ids,
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap();

        (sirrra_program.program, db)
    }

    fn debug_info(db: &RootDatabase, program: &Program) -> DebugInfo {
        // Extract debug information from the program
        DebugInfo::extract(db, program).unwrap()
    }

    #[rstest]
    fn test_extract_debug_info(program: (Program, RootDatabase)) {
        let dbg_info = debug_info(&program.1, &program.0);
        // Assert the debug information contains u128
        assert!(dbg_info
            .type_declarations
            .iter()
            .any(|(k, _)| k.debug_name == Some("u128".into())));

        // Assert the debug information contains u128_sqrt
        assert!(dbg_info
            .libfunc_declarations
            .iter()
            .any(|(k, _)| k.debug_name == Some("u128_sqrt".into())));

        assert!(!dbg_info.statements.is_empty());

        // Assert the debug information contains the run_test function
        assert!(dbg_info.funcs.iter().any(|(k, _)| k
            .debug_name
            .clone()
            .unwrap()
            .contains("run_test")));
    }

    #[rstest]
    fn test_extract_debug_locations(program: (Program, RootDatabase)) {
        // Get the native context
        let native_context = NativeContext::new();

        let dbg_info = debug_info(&program.1, &program.0);

        // Extract debug locations from the debug information
        let debug_locations =
            DebugLocations::extract(native_context.context(), &program.1, &dbg_info);

        // Assert the debug locations contain u128
        assert!(debug_locations
            .type_declarations
            .iter()
            .any(|(k, _)| k.debug_name == Some("u128".into())));

        // Assert the debug locations contain u128_sqrt
        assert!(debug_locations
            .libfunc_declarations
            .iter()
            .any(|(k, _)| k.debug_name == Some("u128_sqrt".into())));

        assert!(!debug_locations.statements.is_empty());

        // Assert the debug locations contain the run_test function
        assert!(debug_locations.funcs.iter().any(|(k, _)| k
            .debug_name
            .clone()
            .unwrap()
            .contains("run_test")));
    }
}
