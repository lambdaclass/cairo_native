use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::program::Program;
use std::{fs, path::Path, sync::Arc};

macro_rules! impl_tests {
    ( $( $name:ident ),* $(,)? ) => {
        $(
            #[test]
            fn $name() {
                let program = if Path::new(concat!("../examples/", stringify!($name), ".sierra")).exists() {
                    load_program(Path::new(concat!("../examples/", stringify!($name), ".sierra")))
                } else {
                    load_program(Path::new(concat!("../examples/", stringify!($name), ".cairo")))
                };

                sierra2mlir::compile(&program, false, false, false, 1).expect("Error compiling sierra program");
            }
        )*
    };
}

impl_tests!(
    //bitwise,
    boolean,
    casts,
    destructure,
    //felt_div,
    felt_is_zero,
    fib,
    fib_simple,
    print_test,
    simple,
    simple_enum,
    types,
    uint,
);

fn load_program(input: &Path) -> Program {
    match input.extension().map(|x| x.to_str().unwrap()) {
        Some("cairo") => Arc::try_unwrap(
            cairo_lang_compiler::compile_cairo_project_at_path(
                input,
                CompilerConfig { replace_ids: true, ..Default::default() },
            )
            .unwrap(),
        )
        .unwrap(),
        Some("sierra") => cairo_lang_sierra::ProgramParser::new()
            .parse(fs::read_to_string(input).unwrap().as_str())
            .unwrap(),
        _ => todo!("unknown file extension"),
    }
}
