use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::program::Program;
use std::{fs, path::Path, sync::Arc};

pub fn prepare_programs() -> impl Iterator<Item = (Program, String)> {
    // Fix corelib search algorithms.
    std::env::set_var(
        "CARGO_MANIFEST_DIR",
        std::env::current_dir().unwrap().join("Cargo.toml").to_str().unwrap(),
    );

    let programs = Path::new(file!())
        .with_file_name("programs")
        .read_dir()
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            match path.extension().map(|x| x.to_str().unwrap()) {
                Some("cairo") => Some((
                    compile_sierra_program(&path),
                    format!(
                        "{0}::{0}::main",
                        path.with_extension("").file_name().unwrap().to_str().unwrap()
                    ),
                )),
                _ => None,
            }
        })
        .collect::<Vec<_>>();

    programs.into_iter()
}

pub fn compile_sierra_program(cairo_path: &Path) -> Program {
    let sierra_path = cairo_path.with_extension("sierra");

    // Load if already compiled and up to date (modified after its cairo source).
    /*if sierra_path.exists()
        && sierra_path.metadata().unwrap().modified().unwrap()
            > cairo_path.metadata().unwrap().modified().unwrap()
    {
        let sierra_code = fs::read_to_string(sierra_path).unwrap();
        ProgramParser::new().parse(&sierra_code).unwrap()
    } else*/
    {
        let program_ptr = cairo_lang_compiler::compile_cairo_project_at_path(
            cairo_path,
            CompilerConfig { replace_ids: true, ..Default::default() },
        )
        .expect("Cairo compilation failed");

        let program = Arc::try_unwrap(program_ptr).unwrap();
        fs::write(sierra_path, program.to_string()).unwrap();

        program
    }
}
