use std::sync::Arc;

use cairo_lang_sierra::program::Program;
use walkdir::WalkDir;

pub fn prepare_programs(path: &str) -> Vec<(Arc<Program>, String)> {
    WalkDir::new(path)
        .into_iter()
        .filter_map(|entry| {
            let e = entry.unwrap();
            let path = e.path();
            match path.extension().map(|x| x.to_str().unwrap()) {
                Some("cairo") => Some((
                    cairo_native::utils::cairo_to_sierra(path),
                    path.file_name().unwrap().to_str().unwrap().to_string(),
                )),
                _ => None,
            }
        })
        .collect::<Vec<_>>()
}
