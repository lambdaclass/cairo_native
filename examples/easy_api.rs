#![feature(iter_intersperse)]

use cairo_lang_sierra::extensions::core::{CoreLibfunc, CoreType};
use cairo_native::easy::Error;
use melior::ExecutionEngine;
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

    let name = cairo_native::easy::felt252_short_str("user");

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
) -> Result<(), Box<Error<'de, CoreType, CoreLibfunc, D, S>>>
where
    D: Deserializer<'de>,
    S: Serializer,
{
    // Compile the cairo program to sierra.
    let program = cairo_native::utils::cairo_to_sierra(program_path);
    let function_id = cairo_native::utils::find_function_id(&program, entry_point);

    // Initialize MLIR.
    let context = cairo_native::easy::initialize_mlir();

    // Compile sierra to MLIR
    let (mut module, registry, required_initial_gas) =
        cairo_native::easy::compile_sierra_to_mlir(&context, &program, function_id)?;

    // Lower MLIR to LLVM
    cairo_native::easy::lower_mlir_to_llvm(&context, &mut module)
        .map_err(|e| Error::JitRunner(e.into()))?;

    // Create the JIT engine.
    let engine = ExecutionEngine::new(&module, 3, &[], false);

    #[cfg(feature = "with-runtime")]
    cairo_native::utils::register_runtime_symbols(&engine);

    // Execute the program
    cairo_native::execute::<CoreType, CoreLibfunc, D, S>(
        &engine,
        &registry,
        function_id,
        params,
        returns,
        required_initial_gas,
    )
    .unwrap_or_else(|e| match &*e {
        cairo_native::error::jit_engine::ErrorImpl::DeserializeError(_) => {
            panic!(
                "Expected inputs with signature: ({})",
                registry
                    .get_function(function_id)
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
