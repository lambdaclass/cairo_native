use afl::fuzz;
use arbitrary::{Arbitrary, Unstructured};
use cairo_lang_starknet_classes::contract_class::version_id_from_serialized_sierra_program;
use cairo_native::{
    executor::AotContractExecutor, include_contract, starknet_stub::StubSyscallHandler, OptLevel,
};
use starknet_types_core::felt::Felt;

fn main() {
    let contract = include_contract!("../test_data_artifacts/contracts/cairo_vm/fib.contract.json");

    let program = contract.extract_sierra_program().unwrap();

    let (sierra_version, _) =
        version_id_from_serialized_sierra_program(&contract.sierra_program).unwrap();
    let executor = AotContractExecutor::new(
        &program,
        &contract.entry_points_by_type,
        sierra_version,
        OptLevel::Aggressive,
        None,
    )
    .unwrap();

    fuzz!(|data: &[u8]| {
        let mut u = Unstructured::new(data);
        let selector = Felt::from_bytes_le(&Arbitrary::arbitrary(&mut u).unwrap());

        let length = u8::arbitrary(&mut u).unwrap();
        let input = (0..length)
            .map(|_| Felt::from_bytes_le(&Arbitrary::arbitrary(&mut u).unwrap()))
            .collect::<Vec<_>>();

        match executor.run(
            selector,
            &input,
            u64::MAX,
            None,
            &mut StubSyscallHandler::default(),
        ) {
            Err(cairo_native::error::Error::SelectorNotFound) => (),
            v => {
                v.unwrap();
            }
        };
    });
}
