use crate::common::run_native_starknet_contract;
use cairo_lang_compiler::CompilerConfig;
use cairo_lang_starknet::compile::compile_path;
use cairo_native::starknet_stub::StubSyscallHandler;
use lazy_static::lazy_static;
use std::path::Path;

lazy_static! {
    static ref KECCAK_CONTRACT: cairo_lang_starknet_classes::contract_class::ContractClass = {
        let path = Path::new("tests/tests/starknet/contracts/test_keccak.cairo");

        compile_path(
            path,
            None,
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap()
    };
}

#[test]
fn keccak_test() {
    let contract = &KECCAK_CONTRACT;

    let entry_point = contract.entry_points_by_type.external.first().unwrap();

    let program = contract.extract_sierra_program().unwrap();
    let result = run_native_starknet_contract(
        &program,
        entry_point.function_idx,
        &[],
        &mut StubSyscallHandler::default(),
    );

    assert!(!result.failure_flag);
    assert_eq!(
        result.remaining_gas,
        340282366920938463463374607431767962375
    );
    assert_eq!(result.return_values, vec![1.into()]);
}
