use cairo_native::{sandbox::IsolatedExecutor, values::JitValue};
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
    let sierra_program = (*cairo_native::utils::cairo_to_sierra(program_path)).clone();

    let mut sandbox = IsolatedExecutor::new(Path::new("/Users/edgar/Documents/cairo_sierra_to_mlir/target/debug/cairo-executor")).unwrap();

    let result = sandbox
        .run_program(
            sierra_program,
            vec![JitValue::Felt252(1.into())],
            "echo::echo::main".to_string(),
        )
        .unwrap();
    dbg!(result);
}
