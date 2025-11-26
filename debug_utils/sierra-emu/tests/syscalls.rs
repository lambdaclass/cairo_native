use std::{path::Path, sync::Arc};

use cairo_lang_compiler::{compile_cairo_project_at_path, CompilerConfig};
use cairo_lang_lowering::utils::InliningStrategy;
use cairo_lang_sierra::program::{GenFunction, Program, StatementIdx};
use sierra_emu::{starknet::StubSyscallHandler, ProgramTrace, VirtualMachine};

fn run_syscall(func_name: &str) -> ProgramTrace {
    let path = Path::new("programs/syscalls.cairo");

    let sierra_program = Arc::new(
        compile_cairo_project_at_path(
            path,
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
            InliningStrategy::Default,
        )
        .unwrap(),
    );

    let function = find_entry_point_by_name(&sierra_program, func_name).unwrap();

    let mut vm = VirtualMachine::new(sierra_program.clone());

    let calldata = [];
    let initial_gas = 1000000;

    vm.call_program(function, initial_gas, calldata);

    let syscall_handler = &mut StubSyscallHandler::default();

    vm.run_with_trace(syscall_handler)
}

#[test]
fn test_contract_constructor() {
    run_syscall("syscalls::syscalls::get_execution_info_v2");
}

pub fn find_entry_point_by_idx(
    program: &Program,
    entry_point_idx: usize,
) -> Option<&GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.id == entry_point_idx as u64)
}

pub fn find_entry_point_by_name<'a>(
    program: &'a Program,
    name: &str,
) -> Option<&'a GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_ref().map(|x| x.as_str()) == Some(name))
}
