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
    use super::*;
    use cairo_lang_semantic::test_utils::setup_test_function;
    use cairo_lang_sierra::program::ConcreteLibfuncLongId;
    use cairo_lang_sierra::program::ConcreteTypeLongId;
    use cairo_lang_sierra::program::FunctionSignature;
    use cairo_lang_sierra::program::GenFunction;
    use cairo_lang_sierra::program::LibfuncDeclaration;
    use cairo_lang_sierra::program::StatementIdx;
    use cairo_lang_sierra::program::TypeDeclaration;
    use cairo_lang_sierra_generator::db::SierraGenGroup;

    #[test]
    fn test_extract_debug_locations() {
        // Build the root database with corelib detection
        let db = RootDatabase::builder().detect_corelib().build().unwrap();

        // Setup a test function using the `setup_test_function` utility
        let test_function = setup_test_function(&db, "fn foo(a: felt252) {}", "foo", "").unwrap();
        let function_id = cairo_lang_lowering::ids::ConcreteFunctionWithBodyId::from_semantic(
            &db,
            test_function.concrete_function_id,
        );
        let _ = db.function_with_body_sierra(function_id);

        // Define a dummy program for testing
        let program = Program {
            type_declarations: vec![TypeDeclaration {
                id: "test_id_type_declarations".into(),
                long_id: ConcreteTypeLongId {
                    generic_id: "u128".into(),
                    generic_args: vec![],
                },
                declared_type_info: None,
            }],
            libfunc_declarations: vec![LibfuncDeclaration {
                id: "test_id_libfunc_declarations".into(),
                long_id: ConcreteLibfuncLongId {
                    generic_id: "u128_sqrt".into(),
                    generic_args: vec![],
                },
            }],
            statements: vec![],
            funcs: vec![GenFunction {
                id: FunctionId {
                    id: 0,
                    debug_name: Some("some_name".into()),
                },
                signature: FunctionSignature {
                    ret_types: vec![],
                    param_types: vec![],
                },
                params: vec![],
                entry_point: StatementIdx(0),
            }],
        };

        // Extract debug information from the program
        let res = DebugInfo::extract(&db, &program).unwrap();

        // Assertions to test the extracted debug information
        assert!(res.type_declarations.len() == 1);
        assert!(res
            .type_declarations
            .contains_key(&ConcreteTypeId::from_string("test_id_type_declarations")));

        assert!(res.libfunc_declarations.len() == 1);
        assert!(res
            .libfunc_declarations
            .contains_key(&ConcreteLibfuncId::from_string(
                "test_id_libfunc_declarations"
            )));

        assert!(res.statements.is_empty());

        assert!(res.funcs.len() == 1);
        assert!(res.funcs.contains_key(&FunctionId {
            id: 0,
            debug_name: Some("some_name".into()),
        }));
    }
}
