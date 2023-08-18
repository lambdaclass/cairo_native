#![feature(iter_intersperse)]

use cairo_native::{error::NativeError, NativeContext, NativeExecutor};
use serde::{Deserializer, Serializer};
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

    // Compile and execute the given sierra program, with the inputs and outputs serialized using JSON.
    compile_and_execute(
        Path::new("programs/examples/hello.cairo"),
        "hello::hello::greet",
        json!([name]),
        &mut serde_json::Serializer::new(stdout()),
    )
    .unwrap();

    println!("Cairo program was compiled and executed succesfully.");
}

/// Shortcut to compile and execute a program.
///
/// For short programs this function may suffice, but as the program grows the other interface is
/// preferred since there is some stuff that should be cached, such as the MLIR context and the
/// execution engines for programs that will be run multiple times.
pub fn compile_and_execute<'de, D, S>(
    program_path: &Path,
    entry_point: &str,
    params: D,
    returns: S,
) -> Result<(), NativeError<'de, D, S>>
where
    D: Deserializer<'de>,
    S: Serializer,
{
    // Compile the cairo program to sierra.
    let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);

    let native_context = NativeContext::new();

    let native_program = native_context
        .compile(&sierra_program)
        .map_err(|e| NativeError::CompilerError(e))?;

    let fn_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);
    let required_init_gas = native_program.get_required_init_gas(&fn_id);
    let native_executor = NativeExecutor::new(native_program);

    native_executor
        .execute(&fn_id, params, returns, required_init_gas)
        .unwrap_or_else(|e| match &*e {
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

    Ok(())
}
