use cairo_native::{context::NativeContext, executor::JitNativeExecutor, values::JitValue};
use starknet_types_core::felt::Felt;
use std::path::Path;

fn main() {
    let program_path = Path::new("programs/examples/hello.cairo");

    // Instantiate a Cairo Native MLIR context. This data structure is responsible for the MLIR
    // initialization and compilation of sierra programs into a MLIR module.
    let native_context = NativeContext::new();

    // Compile the cairo program to sierra.
    let (sierra_program, debug_locations) = cairo_native::utils::cairo_to_sierra_with_debug_info(
        native_context.context(),
        program_path,
    )
    .unwrap();

    // Compile the sierra program into a MLIR module.
    let native_program = native_context
        .compile(&sierra_program, Some(debug_locations))
        .unwrap();

    // The parameters of the entry point.
    let params = &[JitValue::Felt252(Felt::from_bytes_be_slice(b"user"))];

    // Find the entry point id by its name.
    let entry_point = "hello::hello::greet";
    let entry_point_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);

    // Instantiate the executor.
    let native_executor = JitNativeExecutor::from_native_module(
        native_context.context(),
        native_program,
        Default::default(),
    );

    // Execute the program.
    let result = native_executor
        .invoke_dynamic(entry_point_id, params, None)
        .unwrap();

    println!("Cairo program was compiled and executed successfully.");
    println!("{:?}", result);
}
