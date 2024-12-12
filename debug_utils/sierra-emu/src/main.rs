use self::args::CmdArgs;
use cairo_lang_sierra::ProgramParser;
use clap::Parser;
use sierra_emu::run_program;
use std::{
    fs::{self, File},
    io::stdout,
    sync::Arc,
};
use tracing::{info, Level};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

mod args;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CmdArgs::parse();

    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .with_max_level(Level::TRACE)
            .finish(),
    )?;

    info!("Loading the Sierra program from disk.");
    let source_code = fs::read_to_string(args.program)?;

    info!("Parsing the Sierra program.");
    let program = Arc::new(
        ProgramParser::new()
            .parse(&source_code)
            .map_err(|e| e.to_string())?,
    );

    info!("Running the program.");
    let trace = run_program(
        program,
        args.entry_point,
        args.args,
        args.available_gas.unwrap_or_default(),
    );

    match args.output {
        Some(path) => serde_json::to_writer(File::create(path)?, &trace)?,
        None => serde_json::to_writer(stdout().lock(), &trace)?,
    };

    Ok(())
}

#[cfg(test)]
mod test {
    use std::path::Path;

    use cairo_lang_compiler::CompilerConfig;
    use cairo_lang_starknet::compile::compile_path;
    use cairo_lang_starknet_classes::contract_class::version_id_from_serialized_sierra_program;
    use sierra_emu::{starknet::StubSyscallHandler, ContractExecutionResult, VirtualMachine};

    #[test]
    fn test_contract() {
        let path = Path::new("programs/hello_starknet.cairo");

        let contract = compile_path(
            path,
            None,
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap();

        let sierra_program = contract.extract_sierra_program().unwrap();

        let (sierra_version, _) =
            version_id_from_serialized_sierra_program(&contract.sierra_program).unwrap();

        let entry_point = contract.entry_points_by_type.external.first().unwrap();

        let mut vm = VirtualMachine::new_starknet(
            sierra_program.clone().into(),
            &contract.entry_points_by_type,
            sierra_version,
        );

        let calldata = [2.into()];
        let initial_gas = 1000000;

        let syscall_handler = &mut StubSyscallHandler::default();

        // Set the VM at the contract entrypoint
        vm.call_contract(
            entry_point.selector.clone().into(),
            initial_gas,
            calldata,
            None,
        );

        // Run all the steps generating a program execution trace. (Not to be confused with a proof trace)
        let _trace = vm.run_with_trace(syscall_handler);

        // let trace_str = serde_json::to_string_pretty(&trace).unwrap();
        // std::fs::write("contract_trace.json", trace_str).unwrap();
    }

    #[test]
    fn test_contract_constructor() {
        let path = Path::new("programs/hello_starknet.cairo");

        let contract = compile_path(
            path,
            None,
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap();

        let (sierra_version, _) =
            version_id_from_serialized_sierra_program(&contract.sierra_program).unwrap();

        let sierra_program = contract.extract_sierra_program().unwrap();

        let entry_point = contract.entry_points_by_type.constructor.first().unwrap();

        let mut vm = VirtualMachine::new_starknet(
            sierra_program.clone().into(),
            &contract.entry_points_by_type,
            sierra_version,
        );

        let calldata = [2.into()];
        let initial_gas = 1000000;

        let syscall_handler = &mut StubSyscallHandler::default();
        vm.call_contract(
            entry_point.selector.clone().into(),
            initial_gas,
            calldata,
            None,
        );

        let trace = vm.run_with_trace(syscall_handler);

        assert!(!syscall_handler.storage.is_empty());

        let result = ContractExecutionResult::from_trace(&trace).unwrap();
        assert!(!result.failure_flag);
        assert_eq!(result.return_values.len(), 0);
        assert_eq!(result.error_msg, None);

        // let trace_str = serde_json::to_string_pretty(&trace).unwrap();
        // std::fs::write("contract_trace.json", trace_str).unwrap();
    }
}
