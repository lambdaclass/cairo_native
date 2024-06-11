use std::{fs, time::Instant};

use cairo_lang_sierra::ids::FunctionId;
use cairo_lang_starknet::compile::compile_path;
use cairo_native::{
    cache::AotProgramCache, context::NativeContext, starknet::DummySyscallHandler,
    utils::find_entry_point_by_idx,
};
use clap::Parser;
use tracing::{info, info_span, trace};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
struct CliArgs {
    /// Amount of iterations to perform
    iterations: u32,
}

fn main() {
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )
    .expect("failed to set global tracing subscriber");

    let cli_args = CliArgs::parse();

    let native_context = NativeContext::new();
    let mut cache = AotProgramCache::new(&native_context);

    let now = Instant::now();
    let (entry_point, program) = generate_initial_program();
    let elapsed = now.elapsed().as_millis();
    trace!("generated test program, took {elapsed}ms");

    for round in 0..cli_args.iterations {
        let _enter_span = info_span!("round");

        if cache.get(&round).is_some() {
            panic!("all keys should be different")
        }

        let now = Instant::now();
        let executor = cache.compile_and_insert(round, &program, cairo_native::OptLevel::None);
        let elapsed = now.elapsed().as_millis();
        trace!("compiled test program, took {elapsed}ms");

        let now = Instant::now();
        let execution_result = executor
            .invoke_contract_dynamic(&entry_point, &[], Some(u128::MAX), DummySyscallHandler)
            .expect("failed to execute contract");
        let elapsed = now.elapsed().as_millis();
        trace!("executed test program, took {elapsed}ms");

        assert!(
            execution_result.failure_flag == false,
            "contract execution had failure flag set"
        );

        info!(
            "Finished round {round} with result {}",
            execution_result.return_values[0]
        );
    }
}

fn generate_initial_program() -> (FunctionId, cairo_lang_sierra::program::Program) {
    let program_str = format!(
        "\
#[starknet::contract]
mod Contract {{
    #[storage]
    struct Storage {{}}

    #[external(v0)]
    fn main(self: @ContractState) -> felt252 {{
        return 252;
    }}
}}
"
    );

    let mut program_file = tempfile::Builder::new()
        .prefix("test_")
        .suffix(".cairo")
        .tempfile()
        .expect("failed to create temporary file for cairo test program");
    fs::write(&mut program_file, program_str).expect("failed to write cairo test file");

    let contract_class = compile_path(program_file.path(), None, Default::default())
        .expect("failed to compile cairo contract");

    let program = contract_class
        .extract_sierra_program()
        .expect("failed to extract sierra program");

    let entry_point_idx = contract_class
        .entry_points_by_type
        .external
        .first()
        .expect("contrat should have at least one entrypoint")
        .function_idx;

    let entry_point = find_entry_point_by_idx(&program, entry_point_idx)
        .expect("failed to find entrypoint")
        .id
        .clone();

    (entry_point, program)
}
