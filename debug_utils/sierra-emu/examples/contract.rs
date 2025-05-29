use std::{error::Error, sync::Arc};

use std::path::Path;

use cairo_lang_compiler::CompilerConfig;
use cairo_lang_starknet::compile::compile_path;
use cairo_lang_starknet_classes::contract_class::version_id_from_serialized_sierra_program;
use sierra_emu::{starknet::StubSyscallHandler, ContractExecutionResult, VirtualMachine};
use starknet_crypto::Felt;

fn main() -> Result<(), Box<dyn Error>> {
    // INPUT ARGUMENTS
    let cairo_path = Path::new("programs/fibonacci_contract.cairo");
    let arguments: [Felt; 1] = [Felt::from(10)];
    let initial_gas = 100000;

    // Compile cairo to sierra
    let contract = compile_path(
        cairo_path,
        None,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )?;
    let program = contract.extract_sierra_program()?;
    let (version_id, _) = version_id_from_serialized_sierra_program(&contract.sierra_program)?;

    // Find entrypoint to execute
    let entrypoint = contract
        .entry_points_by_type
        .external
        .first()
        .ok_or("contract should contain at least one external entrypoint")?
        .clone();

    // Build virtual machine
    let mut vm = VirtualMachine::new_starknet(
        Arc::new(program),
        &contract.entry_points_by_type,
        version_id,
    );
    vm.call_contract(entrypoint.selector.into(), initial_gas, arguments, None);

    // Execute the virtual machine.
    let syscall_handler = &mut StubSyscallHandler::default();
    let trace = vm.run_with_trace(syscall_handler);

    // Obtain execution result from last frame
    let result =
        ContractExecutionResult::from_trace(&trace).ok_or("failed to obtain execution result")?;

    println!("{result:?}");

    Ok(())
}
