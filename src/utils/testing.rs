use std::{path::Path, sync::Arc};

use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::{program::Program, ProgramParser};

/// Compile a cairo program found at the given path to sierra.
pub fn cairo_to_sierra(program: &Path) -> crate::error::Result<Arc<Program>> {
    if program
        .extension()
        .map(|x| {
            x.to_ascii_lowercase()
                .to_string_lossy()
                .eq_ignore_ascii_case("cairo")
        })
        .unwrap_or(false)
    {
        cairo_lang_compiler::compile_cairo_project_at_path(
            program,
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .map_err(|err| crate::error::Error::ProgramParser(err.to_string()))
    } else {
        let source = std::fs::read_to_string(program)?;
        ProgramParser::new()
            .parse(&source)
            .map_err(|err| crate::error::Error::ProgramParser(err.to_string()))
    }
    .map(Arc::new)
}
