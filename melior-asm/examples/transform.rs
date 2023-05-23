use melior_asm::mlir_asm;
use melior_next::{
    dialect::Registry,
    ir::{Location, Module},
    utility::register_all_dialects,
    Context,
};

fn main() {
    let context = Context::new();

    let registry = Registry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);

    let module = Module::new(Location::unknown(&context));
    let block = module.body();
    mlir_asm! { block, opt("--convert-arith-to-llvm") =>
        func.func @main() -> i32 {
            %0 = arith.constant 0 : i32
            return %0 : i32
        }
    };

    assert!(module.as_operation().verify());
    println!("{}", module.as_operation().debug_print());
}
