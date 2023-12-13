use std::{path::Path, sync::Arc};

use cairo_lang_sierra::program::Program;

pub fn prepare_programs() -> impl Iterator<Item = (Arc<Program>, String)> {
    let programs = Path::new("programs/compile_benches")
        .read_dir()
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            match path.extension().map(|x| x.to_str().unwrap()) {
                Some("cairo") => Some((
                    cairo_native::utils::cairo_to_sierra(&path),
                    path.file_name().unwrap().to_str().unwrap().to_string(),
                )),
                _ => None,
            }
        })
        .collect::<Vec<_>>(); // collect so iter is not lazy evaluated on bench

    programs.into_iter()
}
