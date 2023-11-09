use cairo_lang_sierra::extensions::core::{CoreLibfunc, CoreType, CoreTypeConcrete};
use cairo_lang_sierra::program_registry::ProgramRegistry;
use cairo_native::context::NativeContext;
use cairo_native::executor::NativeExecutor;
use cairo_native::invoke::{InvokeArg, InvokeContext};
use cairo_native::utils::find_entry_point;
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

    let registry: ProgramRegistry<CoreType, CoreLibfunc> =
        ProgramRegistry::new(&sierra_program).unwrap();

    let native_context = NativeContext::new();

    let native_program = native_context.compile(&sierra_program).unwrap();

    // Call the echo function from the contract using the generated wrapper.

    let entry_point_fn = find_entry_point(&sierra_program, "echo::echo::main").unwrap();

    let fn_id = &entry_point_fn.id;
    let required_init_gas = native_program.get_required_init_gas(fn_id);

    /*
    let invoke_context = InvokeContext {
        gas: None,
        system: None,
        args: vec![InvokeArg::Felt252(1.into())],
        ..Default::default()
    };
    */

    let native_executor = NativeExecutor::new(native_program);

    let output = native_executor.execute_args(
        fn_id,
        &[InvokeArg::Felt252(1.into())],
        required_init_gas,
        None,
    );
    dbg!(&output);

    /*
        let result = NativeExecutionResult::deserialize_from_ret_types(
            &mut serde_json::Deserializer::from_slice(&writer),
            &ret_types,
        )
        .expect("failed to serialize starknet execution result");
    */

    println!();
    println!("Cairo program was compiled and executed successfully.");
    println!("{output:#?}");
}
