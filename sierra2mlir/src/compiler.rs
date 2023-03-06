use std::path::Path;

use cairo_lang_sierra::{program::Program, ProgramParser};
use mlir::{context::Context, registry::Registry};



pub struct Compiler {
    pub code: String,
    pub program: Program,
    pub context: Context,
}

impl Compiler {
    pub fn compile_from_code(code: &str) -> color_eyre::Result<()> {
        let code = code.to_string();
        let program: Program = ProgramParser::new().parse(&code).unwrap();

        let context = Context::new();

        let registry = Registry::default();
        registry.register_all_dialects();
        context.append_registry(registry);


        let compiler = Self {
            code,
            program,
            context
        };

        Ok(())
    }
}
