use std::{error::Error, path::Path, sync::Arc};

use cairo_lang_compiler::{compile_cairo_project_at_path, CompilerConfig};
use cairo_lang_lowering::utils::InliningStrategy;
use sierra_emu::{
    find_entry_point_by_name, starknet::StubSyscallHandler, ContractExecutionResult, Value,
    VirtualMachine,
};

fn main() -> Result<(), Box<dyn Error>> {
    // INPUT ARGUMENTS
    let cairo_path = Path::new("programs/fibonacci.cairo");
    let entrypoint = "fibonacci::fibonacci::main";
    let arguments: [Value; 0] = [];
    let initial_gas = 100000;

    // Compile cairo to sierra
    let program = compile_cairo_project_at_path(
        cairo_path,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
        InliningStrategy::Default,
    )?;

    // Find entrypoint to execute
    let function = find_entry_point_by_name(&program, entrypoint)
        .ok_or("failed to find main entrypoint")?
        .clone();

    // Build virtual machine
    let mut vm = VirtualMachine::new(Arc::new(program));
    vm.call_program(&function, initial_gas, arguments);

    // Execute the virtual machine.
    let syscall_handler = &mut StubSyscallHandler::default();
    let trace = vm.run_with_trace(syscall_handler);

    // Obtain execution result from last frame
    let result =
        ContractExecutionResult::from_trace(&trace).ok_or("failed to obtain execution result")?;

    println!("{result:?}");

    Ok(())
}
