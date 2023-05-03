use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::program::Program;
use std::{fs, path::Path, sync::Arc};

macro_rules! impl_tests {
    ( $( $name:ident ),* $(,)? ) => {
        $(
            #[test]
            fn $name() {
                let program = if Path::new(concat!("../examples/", stringify!($name), ".sierra")).exists() {
                    load_program(Path::new(concat!(std::env!("CARGO_MANIFEST_DIR"), "/../examples/", stringify!($name), ".sierra")))
                } else {
                    load_program(Path::new(concat!(std::env!("CARGO_MANIFEST_DIR"), "/../examples/", stringify!($name), ".cairo")))
                };

                sierra2mlir::compile(&program, false, false, false, 1, None).expect("Error compiling sierra program");
            }
        )*
    };
}

impl_tests!(
    bitwise,
    boolean,
    casts,
    destructure,
    enum_match,
    example_array,
    felt_div,
    felt_is_zero,
    fib_simple,
    fib,
    index_array,
    pedersen,
    print,
    print_test,
    simple_enum,
    simple,
    types,
    uint,
    uint_addition
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
