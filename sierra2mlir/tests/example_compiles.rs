macro_rules! impl_tests {
    ( $( $name:ident ),* $(,)? ) => {
        $(
            #[test]
            fn $name() {
                use cairo_lang_sierra::ProgramParser;
                use std::fs::read_to_string;

                let program_path = concat!("../examples/", stringify!($name), ".sierra");
                let sierra_source =
                    read_to_string(program_path).expect("Could not read Sierra source code");

                let program = ProgramParser::new().parse(&sierra_source).unwrap();
                sierra2mlir::compile(&program, false, false, false, 1).expect("Error compiling sierra program");
            }
        )*
    };
}

impl_tests!(
    example_array,
    boolean,
    casts,
    destructure,
    felt_is_zero,
    fib,
    fib_simple,
    print_test,
    program,
    simple,
    simple_enum,
    types,
    uint,
);
