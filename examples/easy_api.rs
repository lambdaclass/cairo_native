#![feature(iter_intersperse)]

use cairo_native::{NativeContext, NativeExecutor};
use serde_json::json;
use std::{io::stdout, path::Path};

fn main() {
    // FIXME: Remove when cairo adds an easy to use API for setting the corelibs path.
    std::env::set_var(
        "CARGO_MANIFEST_DIR",
        format!("{}/a", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
    );

    #[cfg(not(feature = "with-runtime"))]
    compile_error!("This example requires the `with-runtime` feature to be active.");

    let name = cairo_native::utils::felt252_short_str("user");

    let program_path = Path::new("programs/examples/hello.cairo");
    let entry_point = "hello::hello::greet";
    let params = json!([name]);
    let returns = &mut serde_json::Serializer::new(stdout());

    // Compile the cairo program to sierra.
    let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);

    // Instantiate a Cairo Native MLIR contex. This data structure is responsible for the
    // MLIR configuration and compilation of sierra programs into a MLIR module.
    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_program = native_context.compile(&sierra_program).unwrap();

    // Get necessary information for the execution of the program from a given entrypoint:
    //   * entrypoint function id
    //   * required initial gas
    let fn_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);
    let required_init_gas = native_program.get_required_init_gas(&fn_id);

    // Instantiate MLIR executor.
    let native_executor = NativeExecutor::new(native_program);

    // Execute the program
    native_executor
        .execute(&fn_id, params, returns, required_init_gas)
        .unwrap_or_else(|e| match &e.source {
            cairo_native::error::jit_engine::ErrorImpl::DeserializeError(_) => {
                let registry = native_executor.get_program_registry();
                panic!(
                    "Expected inputs with signature: ({})",
                    registry
                        .get_function(fn_id)
                        .unwrap()
                        .signature
                        .param_types
                        .iter()
                        .map(ToString::to_string)
                        .intersperse_with(|| ", ".to_string())
                        .collect::<String>()
                )
            }
            e => panic!("{:?}", e),
        });

    println!("Cairo program was compiled and executed succesfully.");
}
