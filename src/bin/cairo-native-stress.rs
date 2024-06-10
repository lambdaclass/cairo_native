use std::fs;

use cairo_lang_sierra::ids::FunctionId;
use cairo_lang_starknet::compile::compile_path;
use cairo_native::{
    cache::AotProgramCache, context::NativeContext, starknet::DummySyscallHandler,
    utils::find_entry_point_by_idx,
};
use clap::Parser;

#[derive(Parser, Debug)]
struct CliArgs {
    /// Amount of iterations to perform
    iterations: u32,
}

fn main() {
    let cli_args = CliArgs::parse();

    let native_context = NativeContext::new();
    let mut cache = AotProgramCache::new(&native_context);

    for round in 0..cli_args.iterations {
        let (entry_point_id, program) = generate_program("Name", round);

        if cache.get(&round).is_some() {
            panic!("encountered cache hit, all contracts must be different")
        }

        let executor = cache.compile_and_insert(round, &program, cairo_native::OptLevel::None);

        let execution_result = executor
            .invoke_contract_dynamic(&entry_point_id, &[], Some(u128::MAX), DummySyscallHandler)
            .expect("contract execution failed");

        assert!(
            execution_result.failure_flag == false,
            "contract execution failed"
        );

        println!(
            "Finished round {round} with result {}",
            execution_result.return_values[0]
        );
    }
}

fn generate_program(name: &str, output: u32) -> (FunctionId, cairo_lang_sierra::program::Program) {
    let program_str = format!(
        "\
#[starknet::contract]
mod {name} {{
    #[storage]
    struct Storage {{}}

    #[external(v0)]
    fn main(self: @ContractState) -> felt252 {{
        return {output};
    }}
}}
"
    );

    let mut program_file = tempfile::Builder::new()
        .prefix("test_")
        .suffix(".cairo")
        .tempfile()
        .expect("temporary file creation failed");
    fs::write(&mut program_file, program_str).expect("writing to temporary file failed");

    let contract_class = compile_path(program_file.path(), None, Default::default())
        .expect("compiling contract failed");

    let program = contract_class
        .extract_sierra_program()
        .expect("extracting sierra failed");

    let entry_point_id = find_entry_point_by_idx(&program, 0)
        .expect("cairo file should have an entry point")
        .id
        .clone();

    (entry_point_id, program)
}
