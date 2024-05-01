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
use libfuzzer_sys::{
    arbitrary::{Arbitrary, Unstructured},
    fuzz_target,
};
use starknet_types_core::felt::Felt;
use std::path::Path;
use utils::{setup_program, SyscallHandler};

mod utils;

fuzz_target!(|data: (&[u8], Felt)| {
    let native_executor = setup_program();

    //TODO: fuzz point
    let params = &[JitValue::Felt252()];

    let result = native_executor
        .execute_contract(fn_id, params)
        .unwrap();

    result
});
