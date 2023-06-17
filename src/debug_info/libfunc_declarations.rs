use cairo_lang_compiler::db::RootDatabase;
use cairo_lang_defs::{db::DefsGroup, diagnostic_utils::StableLocation, ids::LanguageElementId};
use cairo_lang_diagnostics::DiagnosticAdded;
use cairo_lang_filesystem::db::FilesGroup;
use cairo_lang_sierra::program::LibfuncDeclaration;

pub fn find_libfunc_declaration(
    db: &RootDatabase,
    libfunc_declaration: &LibfuncDeclaration,
) -> Result<Option<StableLocation>, DiagnosticAdded> {
    let libfunc_id = libfunc_declaration.long_id.generic_id.0.as_str();

    for crate_id in db.crates() {
        for module_id in db.crate_modules(crate_id).iter().copied() {
            for extern_libfunc_id in db.module_extern_functions_ids(module_id)? {
                if extern_libfunc_id.name(db) == libfunc_id {
                    return Ok(Some(StableLocation {
                        module_file_id: extern_libfunc_id.module_file_id(db),
                        stable_ptr: extern_libfunc_id.untyped_stable_ptr(db),
                    }));
                }
            }
        }
    }

    Ok(None)
}
