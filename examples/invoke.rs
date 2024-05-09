use cairo_native::{
    context::NativeContext, executor::JitNativeExecutor, utils::find_entry_point, values::JitValue,
};
use std::path::Path;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

fn main() {
    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )
    .unwrap();

    let program_path = Path::new("programs/echo.cairo");

    // Compile the cairo program to sierra.
    let sierra_program = cairo_native::utils::cairo_to_sierra(program_path);

    let native_context = NativeContext::new();

    let native_program = native_context.compile(&sierra_program, None).unwrap();

    // Call the echo function from the contract using the generated wrapper.

    let entry_point_fn = find_entry_point(&sierra_program, "echo::echo::main").unwrap();

    let fn_id = &entry_point_fn.id;

    let native_executor = JitNativeExecutor::from_native_module(
        native_context.context(),
        native_program,
        Default::default(),
    );

    let output = native_executor.invoke_dynamic(fn_id, &[JitValue::Felt252(1.into())], None);

    println!();
    println!("Cairo program was compiled and executed successfully.");
    println!("{output:#?}");
}
