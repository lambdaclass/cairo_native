use self::args::CmdArgs;
use cairo_lang_sierra::{
    extensions::{
        circuit::CircuitTypeConcrete, core::CoreTypeConcrete, starknet::StarknetTypeConcrete,
    },
    ProgramParser,
};
use clap::Parser;
use sierra_emu::{starknet::StubSyscallHandler, Value, VirtualMachine};
use std::{
    fs::{self, File},
    io::stdout,
    sync::Arc,
};
use tracing::{debug, info, Level};
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

    info!("Preparing the virtual machine.");
    let mut vm = VirtualMachine::new(program.clone());

    debug!("Pushing the entry point's frame.");
    let function = program
        .funcs
        .iter()
        .find(|f| match &args.entry_point {
            args::EntryPoint::Number(x) => f.id.id == *x,
            args::EntryPoint::String(x) => f.id.debug_name.as_deref() == Some(x.as_str()),
        })
        .unwrap();

    debug!(
        "Entry point argument types: {:?}",
        function.signature.param_types
    );
    let mut iter = args.args.into_iter();
    vm.push_frame(
        function.id.clone(),
        function
            .signature
            .param_types
            .iter()
            .map(|type_id| {
                let type_info = vm.registry().get_type(type_id).unwrap();
                match type_info {
                    CoreTypeConcrete::Felt252(_) => Value::parse_felt(&iter.next().unwrap()),
                    CoreTypeConcrete::GasBuiltin(_) => Value::U64(args.available_gas.unwrap()),
                    CoreTypeConcrete::RangeCheck(_)
                    | CoreTypeConcrete::RangeCheck96(_)
                    | CoreTypeConcrete::Bitwise(_)
                    | CoreTypeConcrete::Pedersen(_)
                    | CoreTypeConcrete::Poseidon(_)
                    | CoreTypeConcrete::SegmentArena(_)
                    | CoreTypeConcrete::Circuit(
                        CircuitTypeConcrete::AddMod(_) | CircuitTypeConcrete::MulMod(_),
                    ) => Value::Unit,
                    CoreTypeConcrete::Starknet(inner) => match inner {
                        StarknetTypeConcrete::System(_) => Value::Unit,
                        _ => todo!(),
                    },
                    _ => todo!(),
                }
            })
            .collect::<Vec<_>>(),
    );

    info!("Running the program.");
    let syscall_handler = &mut StubSyscallHandler::default();
    let trace = vm.run_with_trace(syscall_handler);

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
