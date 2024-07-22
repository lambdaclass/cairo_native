use cairo_lang_compiler::db::RootDatabase;
use cairo_lang_defs::{diagnostic_utils::StableLocation, ids::LanguageElementId};
use cairo_lang_diagnostics::DiagnosticAdded;
use cairo_lang_sierra::program::Function;
use cairo_lang_sierra_generator::db::SierraGenGroup;

pub fn find_func(
    db: &RootDatabase,
    function: &Function,
) -> Result<StableLocation, DiagnosticAdded> {
    let function_id = db.lookup_intern_sierra_function(function.id.clone());
    let function_with_body_id = function_id
        .body(db)?
        .unwrap()
        .base_semantic_function(db)
        .function_with_body_id(db);

    Ok(function_with_body_id.stable_location(db))
}
