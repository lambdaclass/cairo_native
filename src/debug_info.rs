//! Extracts useful debugging information from cairo programs to make it available to the generated MLIR.

use self::{
    funcs::find_func, libfunc_declarations::find_libfunc_declaration,
    statements::find_all_statements, type_declarations::find_type_declaration,
};
use cairo_lang_compiler::db::RootDatabase;
use cairo_lang_defs::diagnostic_utils::StableLocation;
use cairo_lang_diagnostics::DiagnosticAdded;
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
    pub fn get_func_loc(&self, id: &FunctionId, context: &'c Context) -> Location<'c> {
        if let Some(location) = self.funcs.get(id) {
            *location
        } else {
            Location::unknown(context)
        }
    }

    pub fn get_libfunc_loc(&self, id: &ConcreteLibfuncId, context: &'c Context) -> Location<'c> {
        if let Some(location) = self.libfunc_declarations.get(id) {
            *location
        } else {
            Location::unknown(context)
        }
    }

    pub fn get_type_loc(&self, id: &ConcreteTypeId, context: &'c Context) -> Location<'c> {
        if let Some(location) = self.type_declarations.get(id) {
            *location
        } else {
            Location::unknown(context)
        }
    }

    pub fn get_statement_loc(&self, id: &StatementIdx, context: &'c Context) -> Location<'c> {
        if let Some(location) = self.statements.get(id) {
            *location
        } else {
            Location::unknown(context)
        }
    }

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
                    extract_location_from_stable_loc(
                        context,
                        db,
                        location_id.get(db).stable_location,
                    ),
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

    let path = diagnostic_location.file_id.full_path(db);

    let pos = diagnostic_location
        .span
        .start
        .position_in_file(db, diagnostic_location.file_id)
        .unwrap();

    Location::new(context, &path, pos.line + 1, pos.col + 1)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{context::NativeContext, utils::test::load_cairo};
    use cairo_lang_semantic::test_utils::setup_test_function;
    use cairo_lang_sierra_generator::db::SierraGenGroup;
    use rstest::*;

    #[fixture]
    fn db() -> RootDatabase {
        // Build the root database with corelib detection
        let db = RootDatabase::builder().detect_corelib().build().unwrap();

        // Setup a test function using the `setup_test_function` utility
        let test_function = setup_test_function(&db, "fn foo(a: felt252) {}", "foo", "").unwrap();
        let function_id = cairo_lang_lowering::ids::ConcreteFunctionWithBodyId::from_semantic(
            &db,
            test_function.concrete_function_id,
        );
        let _ = db.function_with_body_sierra(function_id);

        db
    }

    #[fixture]
    fn program() -> Program {
        // Define a dummy program for testing
        let (_, program) = load_cairo! {
            fn run_test() -> u128 {
                let a: u128 = 1;
                u128_sqrt(a).into()
            }
        };
        program
    }

    #[fixture]
    fn debug_info(db: RootDatabase, program: Program) -> DebugInfo {
        // Extract debug information from the program
        DebugInfo::extract(&db, &program).unwrap()
    }

    #[rstest]
    fn test_extract_debug_info(debug_info: DebugInfo) {
        // Assert the debug information contains u128
        assert!(debug_info
            .type_declarations
            .iter()
            .any(|(k, _)| k.debug_name == Some("u128".into())));

        // Assert the debug information contains u128_sqrt
        assert!(debug_info
            .libfunc_declarations
            .iter()
            .any(|(k, _)| k.debug_name == Some("u128_sqrt".into())));

        assert!(debug_info.statements.is_empty());

        // Assert the debug information contains the run_test function
        assert!(debug_info.funcs.iter().any(|(k, _)| k
            .debug_name
            .clone()
            .unwrap()
            .contains("run_test")));
    }

    #[rstest]
    fn test_extract_debug_locations(db: RootDatabase, debug_info: DebugInfo) {
        // Get the native context
        let native_context = NativeContext::new();

        // Extract debug locations from the debug information
        let debug_locations = DebugLocations::extract(native_context.context(), &db, &debug_info);

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

        assert!(debug_locations.statements.is_empty());

        // Assert the debug locations contain the run_test function
        assert!(debug_locations.funcs.iter().any(|(k, _)| k
            .debug_name
            .clone()
            .unwrap()
            .contains("run_test")));
    }
}
