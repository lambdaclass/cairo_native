use afl::fuzz;
use arbitrary::{Arbitrary, Unstructured};
use cairo_lang_starknet_classes::contract_class::{
    version_id_from_serialized_sierra_program, ContractClass,
};
use cairo_native::{executor::AotContractExecutor, starknet_stub::StubSyscallHandler, OptLevel};
use clap::Parser;
use starknet_types_core::felt::Felt;
use std::{fs::File, path::PathBuf};

#[derive(Parser, Debug)]
struct Args {
    contract_path: PathBuf,
}

fn main() {
    let args = Args::parse();

    let contract_file = File::open(args.contract_path).unwrap();
    let contract: ContractClass = serde_json::from_reader(contract_file).unwrap();

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
