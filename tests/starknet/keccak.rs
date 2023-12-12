use cairo_lang_compiler::CompilerConfig;
use cairo_lang_starknet::contract_class::{compile_path, ContractClass};
use cairo_native::starknet::{
    BlockInfo, ExecutionInfo, StarkNetSyscallHandler, SyscallResult, TxInfo, U256,
};
use lazy_static::lazy_static;
use starknet_types_core::felt::Felt as Felt252;
use std::path::Path;

use crate::common::run_native_starknet_contract;

#[derive(Debug)]
struct SyscallHandler;

impl StarkNetSyscallHandler for SyscallHandler {
    fn get_block_hash(&mut self, block_number: u64, _gas: &mut u128) -> SyscallResult<Felt252> {
        println!("Called `get_block_hash({block_number})` from MLIR.");
        Ok(Felt252::from_bytes_be_slice(b"get_block_hash ok"))
    }

    fn get_execution_info(
        &mut self,
        _gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
        println!("Called `get_execution_info()` from MLIR.");
        Ok(ExecutionInfo {
            block_info: BlockInfo {
                block_number: 1234,
                block_timestamp: 2345,
                sequencer_address: 3456.into(),
            },
            tx_info: TxInfo {
                version: 4567.into(),
                account_contract_address: 5678.into(),
                max_fee: 6789,
                signature: vec![1248.into(), 2486.into()],
                transaction_hash: 9876.into(),
                chain_id: 8765.into(),
                nonce: 7654.into(),
            },
            caller_address: 6543.into(),
            contract_address: 5432.into(),
            entry_point_selector: 4321.into(),
        })
    }

    fn deploy(
        &mut self,
        class_hash: Felt252,
        contract_address_salt: Felt252,
        calldata: &[Felt252],
        deploy_from_zero: bool,
        _gas: &mut u128,
    ) -> SyscallResult<(Felt252, Vec<Felt252>)> {
        println!("Called `deploy({class_hash}, {contract_address_salt}, {calldata:?}, {deploy_from_zero})` from MLIR.");
        Ok((
            class_hash + contract_address_salt,
            calldata.iter().map(|x| x + &Felt252::from(1)).collect(),
        ))
    }

    fn replace_class(&mut self, class_hash: Felt252, _gas: &mut u128) -> SyscallResult<()> {
        println!("Called `replace_class({class_hash})` from MLIR.");
        Ok(())
    }

    fn library_call(
        &mut self,
        class_hash: Felt252,
        function_selector: Felt252,
        calldata: &[Felt252],
        _gas: &mut u128,
    ) -> SyscallResult<Vec<Felt252>> {
        println!(
            "Called `library_call({class_hash}, {function_selector}, {calldata:?})` from MLIR."
        );
        Ok(calldata.iter().map(|x| x * &Felt252::from(3)).collect())
    }

    fn call_contract(
        &mut self,
        address: Felt252,
        entry_point_selector: Felt252,
        calldata: &[Felt252],
        _gas: &mut u128,
    ) -> SyscallResult<Vec<Felt252>> {
        println!(
            "Called `call_contract({address}, {entry_point_selector}, {calldata:?})` from MLIR."
        );
        Ok(calldata.iter().map(|x| x * &Felt252::from(3)).collect())
    }

    fn storage_read(
        &mut self,
        address_domain: u32,
        address: Felt252,
        _gas: &mut u128,
    ) -> SyscallResult<Felt252> {
        println!("Called `storage_read({address_domain}, {address})` from MLIR.");
        Ok(address * &Felt252::from(3))
    }

    fn storage_write(
        &mut self,
        address_domain: u32,
        address: Felt252,
        value: Felt252,
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `storage_write({address_domain}, {address}, {value})` from MLIR.");
        Ok(())
    }

    fn emit_event(
        &mut self,
        keys: &[Felt252],
        data: &[Felt252],
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `emit_event({keys:?}, {data:?})` from MLIR.");
        Ok(())
    }

    fn send_message_to_l1(
        &mut self,
        to_address: Felt252,
        payload: &[Felt252],
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `send_message_to_l1({to_address}, {payload:?})` from MLIR.");
        Ok(())
    }

    fn keccak(
        &mut self,
        input: &[u64],
        gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::U256> {
        println!("Called `keccak({input:?})` from MLIR.");
        *gas -= 1000;
        Ok(U256(Felt252::from(1234567890).to_bytes_le()))
    }

    fn secp256k1_add(
        &mut self,
        _p0: cairo_native::starknet::Secp256k1Point,
        _p1: cairo_native::starknet::Secp256k1Point,
        _gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_get_point_from_x(
        &self,
        _x: cairo_native::starknet::U256,
        _y_parity: bool,
        _gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_get_xy(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
        _gas: &mut u128,
    ) -> SyscallResult<(cairo_native::starknet::U256, cairo_native::starknet::U256)> {
        todo!()
    }

    fn secp256k1_mul(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
        _m: cairo_native::starknet::U256,
        _gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_new(
        &self,
        _x: cairo_native::starknet::U256,
        _y: cairo_native::starknet::U256,
        _gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_add(
        &self,
        _p0: cairo_native::starknet::Secp256k1Point,
        _p1: cairo_native::starknet::Secp256k1Point,
        _gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_get_point_from_x(
        &self,
        _x: cairo_native::starknet::U256,
        _y_parity: bool,
        _gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_get_xy(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
        _gas: &mut u128,
    ) -> SyscallResult<(cairo_native::starknet::U256, cairo_native::starknet::U256)> {
        todo!()
    }

    fn secp256r1_mul(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
        _m: cairo_native::starknet::U256,
        _gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_new(
        &mut self,
        _x: cairo_native::starknet::U256,
        _y: cairo_native::starknet::U256,
        _gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn pop_log(&mut self) {
        todo!()
    }

    fn set_account_contract_address(&mut self, _contract_address: Felt252) {
        todo!()
    }

    fn set_block_number(&mut self, _block_number: u64) {
        todo!()
    }

    fn set_block_timestamp(&mut self, _block_timestamp: u64) {
        todo!()
    }

    fn set_caller_address(&mut self, _address: Felt252) {
        todo!()
    }

    fn set_chain_id(&mut self, _chain_id: Felt252) {
        todo!()
    }

    fn set_contract_address(&mut self, _address: Felt252) {
        todo!()
    }

    fn set_max_fee(&mut self, _max_fee: u128) {
        todo!()
    }

    fn set_nonce(&mut self, _nonce: Felt252) {
        todo!()
    }

    fn set_sequencer_address(&mut self, _address: Felt252) {
        todo!()
    }

    fn set_signature(&mut self, _signature: &[Felt252]) {
        todo!()
    }

    fn set_transaction_hash(&mut self, _transaction_hash: Felt252) {
        todo!()
    }

    fn set_version(&mut self, _version: Felt252) {
        todo!()
    }
}

lazy_static! {
    static ref KECCAK_CONTRACT: ContractClass = {
        let path = Path::new("tests/starknet/contracts/test_keccak.cairo");

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

    let entry_point = contract.entry_points_by_type.external.get(0).unwrap();

    let program = contract.extract_sierra_program().unwrap();
    let result =
        run_native_starknet_contract(&program, entry_point.function_idx, &[], &mut SyscallHandler);

    assert!(!result.failure_flag);
    assert_eq!(result.remaining_gas, 18446744073709422805);
    assert_eq!(result.return_values, vec![1.into()]);
}
