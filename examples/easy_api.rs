use cairo_felt::Felt252;
use cairo_native::context::NativeContext;
use cairo_native::executor::NativeExecutor;
use cairo_native::values::JITValue;
use std::path::Path;

fn main() {
    #[cfg(not(feature = "with-runtime"))]
    compile_error!("This example requires the `with-runtime` feature to be active.");

    let program_path = Path::new("programs/examples/hello.cairo");
    // Compile the cairo program to sierra.
    let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);

    // Instantiate a Cairo Native MLIR context. This data structure is responsible for the MLIR
    // initialization and compilation of sierra programs into a MLIR module.
    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_program = native_context.compile(&sierra_program).unwrap();

    // Get necessary information for the execution of the program from a given entrypoint:
    //   - Entrypoint function id
    //   - Required initial gas
    let params = vec![JITValue::Felt252(Felt252::from_bytes_be(b"user"))];
    let entry_point = "hello::hello::greet";
    let fn_id = cairo_native::utils::find_function_id(&sierra_program, entry_point);
    let required_init_gas = native_program.get_required_init_gas(fn_id);

    // Instantiate MLIR executor.
    let native_executor = NativeExecutor::new(native_program);

    // Execute the program.
    let result = native_executor
        .execute(fn_id, &params, required_init_gas, None)
        .unwrap();

    println!("Cairo program was compiled and executed successfully.");
    println!("{:?}", result);
}
