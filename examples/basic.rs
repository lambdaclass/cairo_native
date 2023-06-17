use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ProgramParser,
};
use melior::{dialect::DialectRegistry, utility::register_all_dialects, Context};
use std::{env::args, fs};

fn main() {
    let program = ProgramParser::new()
        .parse(&fs::read_to_string(args().nth(1).unwrap()).unwrap())
        .unwrap();

    let context = Context::new();
    {
        let dialect_registry = DialectRegistry::new();
        register_all_dialects(&dialect_registry);
        context.append_dialect_registry(&dialect_registry);
    }
    context.load_all_available_dialects();

    let module = sierra2mlir::compile::<CoreType, CoreLibfunc>(&context, &program).unwrap();
    println!("{}", module.as_operation());
}
