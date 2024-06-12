////! Extracts useful debugging information from cairo programs to make it available to the generated MLIR.
//! Extracts useful debugging information from cairo programs to make it available to the generated MLIR.
//

//use self::{
use self::{
//    funcs::find_func, libfunc_declarations::find_libfunc_declaration,
    funcs::find_func, libfunc_declarations::find_libfunc_declaration,
//    statements::find_all_statements, type_declarations::find_type_declaration,
    statements::find_all_statements, type_declarations::find_type_declaration,
//};
};
//use cairo_lang_compiler::db::RootDatabase;
use cairo_lang_compiler::db::RootDatabase;
//use cairo_lang_defs::diagnostic_utils::StableLocation;
use cairo_lang_defs::diagnostic_utils::StableLocation;
//use cairo_lang_diagnostics::DiagnosticAdded;
use cairo_lang_diagnostics::DiagnosticAdded;
//use cairo_lang_filesystem::{db::FilesGroup, ids::FileLongId};
use cairo_lang_filesystem::{db::FilesGroup, ids::FileLongId};
//use cairo_lang_lowering::ids::LocationId;
use cairo_lang_lowering::ids::LocationId;
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    ids::{ConcreteLibfuncId, ConcreteTypeId, FunctionId},
    ids::{ConcreteLibfuncId, ConcreteTypeId, FunctionId},
//    program::{Program, StatementIdx},
    program::{Program, StatementIdx},
//};
};
//use melior::{ir::Location, Context};
use melior::{ir::Location, Context};
//use std::collections::HashMap;
use std::collections::HashMap;
//

//mod funcs;
mod funcs;
//mod libfunc_declarations;
mod libfunc_declarations;
//mod statements;
mod statements;
//mod type_declarations;
mod type_declarations;
//

//#[derive(Clone, Debug)]
#[derive(Clone, Debug)]
//pub struct DebugInfo {
pub struct DebugInfo {
//    pub type_declarations: HashMap<ConcreteTypeId, StableLocation>,
    pub type_declarations: HashMap<ConcreteTypeId, StableLocation>,
//    pub libfunc_declarations: HashMap<ConcreteLibfuncId, StableLocation>,
    pub libfunc_declarations: HashMap<ConcreteLibfuncId, StableLocation>,
//    pub statements: HashMap<StatementIdx, LocationId>,
    pub statements: HashMap<StatementIdx, LocationId>,
//    pub funcs: HashMap<FunctionId, StableLocation>,
    pub funcs: HashMap<FunctionId, StableLocation>,
//}
}
//

//impl DebugInfo {
impl DebugInfo {
//    pub fn extract(db: &RootDatabase, program: &Program) -> Result<Self, DiagnosticAdded> {
    pub fn extract(db: &RootDatabase, program: &Program) -> Result<Self, DiagnosticAdded> {
//        let type_declarations = program
        let type_declarations = program
//            .type_declarations
            .type_declarations
//            .iter()
            .iter()
//            .filter_map(|type_declaration| {
            .filter_map(|type_declaration| {
//                find_type_declaration(db, type_declaration)
                find_type_declaration(db, type_declaration)
//                    .map(|x| x.map(|location| (type_declaration.id.clone(), location)))
                    .map(|x| x.map(|location| (type_declaration.id.clone(), location)))
//                    .transpose()
                    .transpose()
//            })
            })
//            .collect::<Result<HashMap<_, _>, DiagnosticAdded>>()?;
            .collect::<Result<HashMap<_, _>, DiagnosticAdded>>()?;
//

//        let libfunc_declarations = program
        let libfunc_declarations = program
//            .libfunc_declarations
            .libfunc_declarations
//            .iter()
            .iter()
//            .filter_map(|libfunc_declaration| {
            .filter_map(|libfunc_declaration| {
//                find_libfunc_declaration(db, libfunc_declaration)
                find_libfunc_declaration(db, libfunc_declaration)
//                    .map(|x| x.map(|location| (libfunc_declaration.id.clone(), location)))
                    .map(|x| x.map(|location| (libfunc_declaration.id.clone(), location)))
//                    .transpose()
                    .transpose()
//            })
            })
//            .collect::<Result<HashMap<_, _>, DiagnosticAdded>>()?;
            .collect::<Result<HashMap<_, _>, DiagnosticAdded>>()?;
//

//        let statements =
        let statements =
//            find_all_statements(db, |id| libfunc_declarations.contains_key(id), program)?;
            find_all_statements(db, |id| libfunc_declarations.contains_key(id), program)?;
//

//        let funcs = program
        let funcs = program
//            .funcs
            .funcs
//            .iter()
            .iter()
//            .map(|function| Ok((function.id.clone(), find_func(db, function)?)))
            .map(|function| Ok((function.id.clone(), find_func(db, function)?)))
//            .collect::<Result<HashMap<_, _>, DiagnosticAdded>>()?;
            .collect::<Result<HashMap<_, _>, DiagnosticAdded>>()?;
//

//        Ok(Self {
        Ok(Self {
//            type_declarations,
            type_declarations,
//            libfunc_declarations,
            libfunc_declarations,
//            statements,
            statements,
//            funcs,
            funcs,
//        })
        })
//    }
    }
//}
}
//

//#[derive(Clone, Debug)]
#[derive(Clone, Debug)]
//pub struct DebugLocations<'c> {
pub struct DebugLocations<'c> {
//    pub type_declarations: HashMap<ConcreteTypeId, Location<'c>>,
    pub type_declarations: HashMap<ConcreteTypeId, Location<'c>>,
//    pub libfunc_declarations: HashMap<ConcreteLibfuncId, Location<'c>>,
    pub libfunc_declarations: HashMap<ConcreteLibfuncId, Location<'c>>,
//    pub statements: HashMap<StatementIdx, Location<'c>>,
    pub statements: HashMap<StatementIdx, Location<'c>>,
//    pub funcs: HashMap<FunctionId, Location<'c>>,
    pub funcs: HashMap<FunctionId, Location<'c>>,
//}
}
//

//impl<'c> DebugLocations<'c> {
impl<'c> DebugLocations<'c> {
//    pub fn extract(context: &'c Context, db: &RootDatabase, debug_info: &DebugInfo) -> Self {
    pub fn extract(context: &'c Context, db: &RootDatabase, debug_info: &DebugInfo) -> Self {
//        let type_declarations = debug_info
        let type_declarations = debug_info
//            .type_declarations
            .type_declarations
//            .iter()
            .iter()
//            .map(|(type_id, stable_loc)| {
            .map(|(type_id, stable_loc)| {
//                (
                (
//                    type_id.clone(),
                    type_id.clone(),
//                    extract_location_from_stable_loc(context, db, *stable_loc),
                    extract_location_from_stable_loc(context, db, *stable_loc),
//                )
                )
//            })
            })
//            .collect();
            .collect();
//

//        let libfunc_declarations = debug_info
        let libfunc_declarations = debug_info
//            .libfunc_declarations
            .libfunc_declarations
//            .iter()
            .iter()
//            .map(|(libfunc_id, stable_loc)| {
            .map(|(libfunc_id, stable_loc)| {
//                (
                (
//                    libfunc_id.clone(),
                    libfunc_id.clone(),
//                    extract_location_from_stable_loc(context, db, *stable_loc),
                    extract_location_from_stable_loc(context, db, *stable_loc),
//                )
                )
//            })
            })
//            .collect();
            .collect();
//

//        let statements = debug_info
        let statements = debug_info
//            .statements
            .statements
//            .iter()
            .iter()
//            .map(|(statement_idx, location_id)| {
            .map(|(statement_idx, location_id)| {
//                (
                (
//                    *statement_idx,
                    *statement_idx,
//                    extract_location_from_stable_loc(
                    extract_location_from_stable_loc(
//                        context,
                        context,
//                        db,
                        db,
//                        location_id.get(db).stable_location,
                        location_id.get(db).stable_location,
//                    ),
                    ),
//                )
                )
//            })
            })
//            .collect();
            .collect();
//

//        let funcs = debug_info
        let funcs = debug_info
//            .funcs
            .funcs
//            .iter()
            .iter()
//            .map(|(function_id, stable_loc)| {
            .map(|(function_id, stable_loc)| {
//                (
                (
//                    function_id.clone(),
                    function_id.clone(),
//                    extract_location_from_stable_loc(context, db, *stable_loc),
                    extract_location_from_stable_loc(context, db, *stable_loc),
//                )
                )
//            })
            })
//            .collect();
            .collect();
//

//        Self {
        Self {
//            type_declarations,
            type_declarations,
//            libfunc_declarations,
            libfunc_declarations,
//            statements,
            statements,
//            funcs,
            funcs,
//        }
        }
//    }
    }
//}
}
//

//fn extract_location_from_stable_loc<'c>(
fn extract_location_from_stable_loc<'c>(
//    context: &'c Context,
    context: &'c Context,
//    db: &RootDatabase,
    db: &RootDatabase,
//    stable_loc: StableLocation,
    stable_loc: StableLocation,
//) -> Location<'c> {
) -> Location<'c> {
//    let diagnostic_location = stable_loc.diagnostic_location(db);
    let diagnostic_location = stable_loc.diagnostic_location(db);
//

//    let path = match db.lookup_intern_file(diagnostic_location.file_id) {
    let path = match db.lookup_intern_file(diagnostic_location.file_id) {
//        FileLongId::OnDisk(path) => path,
        FileLongId::OnDisk(path) => path,
//        FileLongId::Virtual(_) => return Location::unknown(context),
        FileLongId::Virtual(_) => return Location::unknown(context),
//    };
    };
//

//    let pos = diagnostic_location
    let pos = diagnostic_location
//        .span
        .span
//        .start
        .start
//        .position_in_file(db, diagnostic_location.file_id)
        .position_in_file(db, diagnostic_location.file_id)
//        .unwrap();
        .unwrap();
//

//    Location::new(context, &path.to_string_lossy(), pos.line, pos.col)
    Location::new(context, &path.to_string_lossy(), pos.line, pos.col)
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use super::*;
    use super::*;
//    use crate::{context::NativeContext, utils::test::load_cairo};
    use crate::{context::NativeContext, utils::test::load_cairo};
//    use cairo_lang_semantic::test_utils::setup_test_function;
    use cairo_lang_semantic::test_utils::setup_test_function;
//    use cairo_lang_sierra_generator::db::SierraGenGroup;
    use cairo_lang_sierra_generator::db::SierraGenGroup;
//    use rstest::*;
    use rstest::*;
//

//    #[fixture]
    #[fixture]
//    fn db() -> RootDatabase {
    fn db() -> RootDatabase {
//        // Build the root database with corelib detection
        // Build the root database with corelib detection
//        let db = RootDatabase::builder().detect_corelib().build().unwrap();
        let db = RootDatabase::builder().detect_corelib().build().unwrap();
//

//        // Setup a test function using the `setup_test_function` utility
        // Setup a test function using the `setup_test_function` utility
//        let test_function = setup_test_function(&db, "fn foo(a: felt252) {}", "foo", "").unwrap();
        let test_function = setup_test_function(&db, "fn foo(a: felt252) {}", "foo", "").unwrap();
//        let function_id = cairo_lang_lowering::ids::ConcreteFunctionWithBodyId::from_semantic(
        let function_id = cairo_lang_lowering::ids::ConcreteFunctionWithBodyId::from_semantic(
//            &db,
            &db,
//            test_function.concrete_function_id,
            test_function.concrete_function_id,
//        );
        );
//        let _ = db.function_with_body_sierra(function_id);
        let _ = db.function_with_body_sierra(function_id);
//

//        db
        db
//    }
    }
//

//    #[fixture]
    #[fixture]
//    fn program() -> Program {
    fn program() -> Program {
//        // Define a dummy program for testing
        // Define a dummy program for testing
//        let (_, program) = load_cairo! {
        let (_, program) = load_cairo! {
//            fn run_test() -> u128 {
            fn run_test() -> u128 {
//                let a: u128 = 1;
                let a: u128 = 1;
//                u128_sqrt(a).into()
                u128_sqrt(a).into()
//            }
            }
//        };
        };
//        program
        program
//    }
    }
//

//    #[fixture]
    #[fixture]
//    fn debug_info(db: RootDatabase, program: Program) -> DebugInfo {
    fn debug_info(db: RootDatabase, program: Program) -> DebugInfo {
//        // Extract debug information from the program
        // Extract debug information from the program
//        DebugInfo::extract(&db, &program).unwrap()
        DebugInfo::extract(&db, &program).unwrap()
//    }
    }
//

//    #[rstest]
    #[rstest]
//    fn test_extract_debug_info(debug_info: DebugInfo) {
    fn test_extract_debug_info(debug_info: DebugInfo) {
//        // Assert the debug information contains u128
        // Assert the debug information contains u128
//        assert!(debug_info
        assert!(debug_info
//            .type_declarations
            .type_declarations
//            .iter()
            .iter()
//            .any(|(k, _)| k.debug_name == Some("u128".into())));
            .any(|(k, _)| k.debug_name == Some("u128".into())));
//

//        // Assert the debug information contains u128_sqrt
        // Assert the debug information contains u128_sqrt
//        assert!(debug_info
        assert!(debug_info
//            .libfunc_declarations
            .libfunc_declarations
//            .iter()
            .iter()
//            .any(|(k, _)| k.debug_name == Some("u128_sqrt".into())));
            .any(|(k, _)| k.debug_name == Some("u128_sqrt".into())));
//

//        assert!(debug_info.statements.is_empty());
        assert!(debug_info.statements.is_empty());
//

//        // Assert the debug information contains the run_test function
        // Assert the debug information contains the run_test function
//        assert!(debug_info.funcs.iter().any(|(k, _)| k
        assert!(debug_info.funcs.iter().any(|(k, _)| k
//            .debug_name
            .debug_name
//            .clone()
            .clone()
//            .unwrap()
            .unwrap()
//            .contains("run_test")));
            .contains("run_test")));
//    }
    }
//

//    #[rstest]
    #[rstest]
//    fn test_extract_debug_locations(db: RootDatabase, debug_info: DebugInfo) {
    fn test_extract_debug_locations(db: RootDatabase, debug_info: DebugInfo) {
//        // Get the native context
        // Get the native context
//        let native_context = NativeContext::new();
        let native_context = NativeContext::new();
//

//        // Extract debug locations from the debug information
        // Extract debug locations from the debug information
//        let debug_locations = DebugLocations::extract(native_context.context(), &db, &debug_info);
        let debug_locations = DebugLocations::extract(native_context.context(), &db, &debug_info);
//

//        // Assert the debug locations contain u128
        // Assert the debug locations contain u128
//        assert!(debug_locations
        assert!(debug_locations
//            .type_declarations
            .type_declarations
//            .iter()
            .iter()
//            .any(|(k, _)| k.debug_name == Some("u128".into())));
            .any(|(k, _)| k.debug_name == Some("u128".into())));
//

//        // Assert the debug locations contain u128_sqrt
        // Assert the debug locations contain u128_sqrt
//        assert!(debug_locations
        assert!(debug_locations
//            .libfunc_declarations
            .libfunc_declarations
//            .iter()
            .iter()
//            .any(|(k, _)| k.debug_name == Some("u128_sqrt".into())));
            .any(|(k, _)| k.debug_name == Some("u128_sqrt".into())));
//

//        assert!(debug_locations.statements.is_empty());
        assert!(debug_locations.statements.is_empty());
//

//        // Assert the debug locations contain the run_test function
        // Assert the debug locations contain the run_test function
//        assert!(debug_locations.funcs.iter().any(|(k, _)| k
        assert!(debug_locations.funcs.iter().any(|(k, _)| k
//            .debug_name
            .debug_name
//            .clone()
            .clone()
//            .unwrap()
            .unwrap()
//            .contains("run_test")));
            .contains("run_test")));
//    }
    }
//}
}
