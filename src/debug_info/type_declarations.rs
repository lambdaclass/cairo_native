use cairo_lang_compiler::db::RootDatabase;
use cairo_lang_defs::{
    db::DefsGroup,
    diagnostic_utils::StableLocation,
    ids::{LanguageElementId, NamedLanguageElementId},
};
use cairo_lang_diagnostics::DiagnosticAdded;
use cairo_lang_filesystem::db::FilesGroup;
use cairo_lang_sierra::program::TypeDeclaration;

pub fn find_type_declaration(
    db: &RootDatabase,
    type_declaration: &TypeDeclaration,
) -> Result<Option<StableLocation>, DiagnosticAdded> {
    let type_id = type_declaration.long_id.generic_id.0.as_str();

    for crate_id in db.crates() {
        for module_id in db.crate_modules(crate_id).iter().copied() {
            for extern_type_id in db.module_extern_types_ids(module_id)?.iter() {
                if extern_type_id.name(db) == type_id {
                    return Ok(Some(StableLocation::new(
                        extern_type_id.untyped_stable_ptr(db),
                    )));
                }
            }
        }
    }

    Ok(None)
}
