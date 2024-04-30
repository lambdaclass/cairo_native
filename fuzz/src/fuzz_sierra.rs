#![no_main]
use cairo_lang_compiler::CompilerConfig;
use cairo_lang_starknet::contract_class::compile_path;
use cairo_native::{
    context::NativeContext,
    executor::NativeExecutor,
    metadata::syscall_handler::SyscallHandlerMeta,
    starknet::{BlockInfo, ExecutionInfo, StarknetSyscallHandler, SyscallResult, TxInfo, U256},
    utils::find_entry_point_by_idx,
    values::JitValue,
};
use libfuzzer_sys::fuzz_target;
use starknet_types_core::felt::Felt;
use utils::SyscallHandler;
use std::path::Path;

mod utils;


fuzz_target!(|data: (&[u8], JitValue)| {
    // fuzzed code goes here
});


fn fuzz_sierra_program(program_path: &str) {
    let path = Path::new(program_path);

    let contract = compile_path(
        path,
        None,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )?;

    let entry_point = contract.entry_points_by_type.constructor.get(0)?;
    let sierra_program = contract.extract_sierra_program()?;
    let native_context = NativeContext::new();

    let mut native_program = native_context.compile(&sierra_program).unwrap();
    native_program
        .insert_metadata(SyscallHandlerMeta::new(&mut SyscallHandler))
        .unwrap();

    // Call the echo function from the contract using the generated wrapper.
    let entry_point_fn =
        find_entry_point_by_idx(&sierra_program, entry_point.function_idx).unwrap();

    let fn_id = &entry_point_fn.id;

    let native_executor = NativeExecutor::new(native_program);

    // fuzz this
    let params = &[JitValue::Felt252(Felt::from(1))];

    let result = native_executor
        .execute_contract(fn_id, params, u64::MAX.into())
        .expect("failed to execute the given contract");
}
