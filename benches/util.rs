//use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_runner::SierraCasmRunner;
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use std::sync::Arc;
use std::sync::Arc;
//use walkdir::WalkDir;
use walkdir::WalkDir;
//

//pub fn prepare_programs(path: &str) -> Vec<(Arc<Program>, String)> {
pub fn prepare_programs(path: &str) -> Vec<(Arc<Program>, String)> {
//    WalkDir::new(path)
    WalkDir::new(path)
//        .into_iter()
        .into_iter()
//        .filter_map(|entry| {
        .filter_map(|entry| {
//            let e = entry.unwrap();
            let e = entry.unwrap();
//            let path = e.path();
            let path = e.path();
//            match path.extension().map(|x| x.to_str().unwrap()) {
            match path.extension().map(|x| x.to_str().unwrap()) {
//                Some("cairo") => Some((
                Some("cairo") => Some((
//                    cairo_native::utils::cairo_to_sierra(path),
                    cairo_native::utils::cairo_to_sierra(path),
//                    path.display().to_string(),
                    path.display().to_string(),
//                )),
                )),
//                _ => None,
                _ => None,
//            }
            }
//        })
        })
//        .collect::<Vec<_>>()
        .collect::<Vec<_>>()
//}
}
//

//#[allow(unused)] // its used but clippy doesn't detect it well
#[allow(unused)] // its used but clippy doesn't detect it well
//pub fn create_vm_runner(program: &Program) -> SierraCasmRunner {
pub fn create_vm_runner(program: &Program) -> SierraCasmRunner {
//    SierraCasmRunner::new(
    SierraCasmRunner::new(
//        program.clone(),
        program.clone(),
//        Some(Default::default()),
        Some(Default::default()),
//        Default::default(),
        Default::default(),
//        None,
        None,
//    )
    )
//    .unwrap()
    .unwrap()
//}
}
