use std::path::Path;

use cairo_lang_starknet_classes::contract_class::ContractClass;
use cairo_native::{
    context::NativeContext, executor::JitNativeExecutor, starknet_stub::StubSyscallHandler,
    utils::find_entry_point_by_idx,
};

#[test]
pub fn test_oz_erc20() {
    let path = Path::new("tests/erc20/target/dev/native_erc20_Native.contract_class.json");
    let sierra_json = std::fs::read_to_string(path)
        .expect("failed to read native_erc20.sierra.json, please run make build-erc20");
    let contract: ContractClass =
        serde_json::from_str(&sierra_json).expect("failed to deserialize program");

    dbg!(&contract.entry_points_by_type.constructor);
    let entry_point = contract.entry_points_by_type.constructor.first().unwrap();
    let sierra_program = contract.extract_sierra_program().unwrap();

    let native_context = NativeContext::new();

    let native_program = native_context.compile(&sierra_program, None).unwrap();

    let entry_point_fn =
        find_entry_point_by_idx(&sierra_program, entry_point.function_idx).unwrap();
    let fn_id = &entry_point_fn.id;

    let native_executor = JitNativeExecutor::from_native_module(native_program, Default::default());

    let mut handler = StubSyscallHandler::default();

    let _result = native_executor
        .invoke_contract_dynamic(
            fn_id,
            &[
                10.into(),  // initial supply, u256 ?
                10.into(),  // initial supply, u256 ?
                0x2.into(), // recipient
            ],
            Some(u128::MAX),
            &mut handler,
        )
        .expect("failed to execute the given contract");
    dbg!(_result);
}
