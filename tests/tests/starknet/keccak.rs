//use crate::common::run_native_starknet_contract;
use crate::common::run_native_starknet_contract;
//use cairo_lang_compiler::CompilerConfig;
use cairo_lang_compiler::CompilerConfig;
//use cairo_lang_starknet::compile::compile_path;
use cairo_lang_starknet::compile::compile_path;
//use cairo_native::starknet::{
use cairo_native::starknet::{
//    BlockInfo, ExecutionInfo, ExecutionInfoV2, ResourceBounds, Secp256k1Point, Secp256r1Point,
    BlockInfo, ExecutionInfo, ExecutionInfoV2, ResourceBounds, Secp256k1Point, Secp256r1Point,
//    StarknetSyscallHandler, SyscallResult, TxInfo, TxV2Info, U256,
    StarknetSyscallHandler, SyscallResult, TxInfo, TxV2Info, U256,
//};
};
//use lazy_static::lazy_static;
use lazy_static::lazy_static;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use std::path::Path;
use std::path::Path;
//

//#[derive(Debug)]
#[derive(Debug)]
//struct SyscallHandler;
struct SyscallHandler;
//

//impl StarknetSyscallHandler for SyscallHandler {
impl StarknetSyscallHandler for SyscallHandler {
//    fn get_block_hash(&mut self, block_number: u64, _gas: &mut u128) -> SyscallResult<Felt> {
    fn get_block_hash(&mut self, block_number: u64, _gas: &mut u128) -> SyscallResult<Felt> {
//        println!("Called `get_block_hash({block_number})` from MLIR.");
        println!("Called `get_block_hash({block_number})` from MLIR.");
//        Ok(Felt::from_bytes_be_slice(b"get_block_hash ok"))
        Ok(Felt::from_bytes_be_slice(b"get_block_hash ok"))
//    }
    }
//

//    fn get_execution_info(
    fn get_execution_info(
//        &mut self,
        &mut self,
//        _gas: &mut u128,
        _gas: &mut u128,
//    ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
//        println!("Called `get_execution_info()` from MLIR.");
        println!("Called `get_execution_info()` from MLIR.");
//        Ok(ExecutionInfo {
        Ok(ExecutionInfo {
//            block_info: BlockInfo {
            block_info: BlockInfo {
//                block_number: 1234,
                block_number: 1234,
//                block_timestamp: 2345,
                block_timestamp: 2345,
//                sequencer_address: 3456.into(),
                sequencer_address: 3456.into(),
//            },
            },
//            tx_info: TxInfo {
            tx_info: TxInfo {
//                version: 4567.into(),
                version: 4567.into(),
//                account_contract_address: 5678.into(),
                account_contract_address: 5678.into(),
//                max_fee: 6789,
                max_fee: 6789,
//                signature: vec![1248.into(), 2486.into()],
                signature: vec![1248.into(), 2486.into()],
//                transaction_hash: 9876.into(),
                transaction_hash: 9876.into(),
//                chain_id: 8765.into(),
                chain_id: 8765.into(),
//                nonce: 7654.into(),
                nonce: 7654.into(),
//            },
            },
//            caller_address: 6543.into(),
            caller_address: 6543.into(),
//            contract_address: 5432.into(),
            contract_address: 5432.into(),
//            entry_point_selector: 4321.into(),
            entry_point_selector: 4321.into(),
//        })
        })
//    }
    }
//

//    fn get_execution_info_v2(
    fn get_execution_info_v2(
//        &mut self,
        &mut self,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<cairo_native::starknet::ExecutionInfoV2> {
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfoV2> {
//        println!("Called `get_execution_info_v2()` from MLIR.");
        println!("Called `get_execution_info_v2()` from MLIR.");
//        Ok(ExecutionInfoV2 {
        Ok(ExecutionInfoV2 {
//            block_info: BlockInfo {
            block_info: BlockInfo {
//                block_number: 1234,
                block_number: 1234,
//                block_timestamp: 2345,
                block_timestamp: 2345,
//                sequencer_address: 3456.into(),
                sequencer_address: 3456.into(),
//            },
            },
//            tx_info: TxV2Info {
            tx_info: TxV2Info {
//                version: 1.into(),
                version: 1.into(),
//                account_contract_address: 1.into(),
                account_contract_address: 1.into(),
//                max_fee: 0,
                max_fee: 0,
//                signature: vec![1.into()],
                signature: vec![1.into()],
//                transaction_hash: 1.into(),
                transaction_hash: 1.into(),
//                chain_id: 1.into(),
                chain_id: 1.into(),
//                nonce: 1.into(),
                nonce: 1.into(),
//                tip: 1,
                tip: 1,
//                paymaster_data: vec![1.into()],
                paymaster_data: vec![1.into()],
//                nonce_data_availability_mode: 0,
                nonce_data_availability_mode: 0,
//                fee_data_availability_mode: 0,
                fee_data_availability_mode: 0,
//                account_deployment_data: vec![1.into()],
                account_deployment_data: vec![1.into()],
//                resource_bounds: vec![ResourceBounds {
                resource_bounds: vec![ResourceBounds {
//                    resource: 2.into(),
                    resource: 2.into(),
//                    max_amount: 10,
                    max_amount: 10,
//                    max_price_per_unit: 20,
                    max_price_per_unit: 20,
//                }],
                }],
//            },
            },
//            caller_address: 6543.into(),
            caller_address: 6543.into(),
//            contract_address: 5432.into(),
            contract_address: 5432.into(),
//            entry_point_selector: 4321.into(),
            entry_point_selector: 4321.into(),
//        })
        })
//    }
    }
//

//    fn deploy(
    fn deploy(
//        &mut self,
        &mut self,
//        class_hash: Felt,
        class_hash: Felt,
//        contract_address_salt: Felt,
        contract_address_salt: Felt,
//        calldata: &[Felt],
        calldata: &[Felt],
//        deploy_from_zero: bool,
        deploy_from_zero: bool,
//        _gas: &mut u128,
        _gas: &mut u128,
//    ) -> SyscallResult<(Felt, Vec<Felt>)> {
    ) -> SyscallResult<(Felt, Vec<Felt>)> {
//        println!("Called `deploy({class_hash}, {contract_address_salt}, {calldata:?}, {deploy_from_zero})` from MLIR.");
        println!("Called `deploy({class_hash}, {contract_address_salt}, {calldata:?}, {deploy_from_zero})` from MLIR.");
//        Ok((
        Ok((
//            class_hash + contract_address_salt,
            class_hash + contract_address_salt,
//            calldata.iter().map(|x| x + Felt::ONE).collect(),
            calldata.iter().map(|x| x + Felt::ONE).collect(),
//        ))
        ))
//    }
    }
//

//    fn replace_class(&mut self, class_hash: Felt, _gas: &mut u128) -> SyscallResult<()> {
    fn replace_class(&mut self, class_hash: Felt, _gas: &mut u128) -> SyscallResult<()> {
//        println!("Called `replace_class({class_hash})` from MLIR.");
        println!("Called `replace_class({class_hash})` from MLIR.");
//        Ok(())
        Ok(())
//    }
    }
//

//    fn library_call(
    fn library_call(
//        &mut self,
        &mut self,
//        class_hash: Felt,
        class_hash: Felt,
//        function_selector: Felt,
        function_selector: Felt,
//        calldata: &[Felt],
        calldata: &[Felt],
//        _gas: &mut u128,
        _gas: &mut u128,
//    ) -> SyscallResult<Vec<Felt>> {
    ) -> SyscallResult<Vec<Felt>> {
//        println!(
        println!(
//            "Called `library_call({class_hash}, {function_selector}, {calldata:?})` from MLIR."
            "Called `library_call({class_hash}, {function_selector}, {calldata:?})` from MLIR."
//        );
        );
//        Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
        Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
//    }
    }
//

//    fn call_contract(
    fn call_contract(
//        &mut self,
        &mut self,
//        address: Felt,
        address: Felt,
//        entry_point_selector: Felt,
        entry_point_selector: Felt,
//        calldata: &[Felt],
        calldata: &[Felt],
//        _gas: &mut u128,
        _gas: &mut u128,
//    ) -> SyscallResult<Vec<Felt>> {
    ) -> SyscallResult<Vec<Felt>> {
//        println!(
        println!(
//            "Called `call_contract({address}, {entry_point_selector}, {calldata:?})` from MLIR."
            "Called `call_contract({address}, {entry_point_selector}, {calldata:?})` from MLIR."
//        );
        );
//        Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
        Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
//    }
    }
//

//    fn storage_read(
    fn storage_read(
//        &mut self,
        &mut self,
//        address_domain: u32,
        address_domain: u32,
//        address: Felt,
        address: Felt,
//        _gas: &mut u128,
        _gas: &mut u128,
//    ) -> SyscallResult<Felt> {
    ) -> SyscallResult<Felt> {
//        println!("Called `storage_read({address_domain}, {address})` from MLIR.");
        println!("Called `storage_read({address_domain}, {address})` from MLIR.");
//        Ok(address * Felt::from(3))
        Ok(address * Felt::from(3))
//    }
    }
//

//    fn storage_write(
    fn storage_write(
//        &mut self,
        &mut self,
//        address_domain: u32,
        address_domain: u32,
//        address: Felt,
        address: Felt,
//        value: Felt,
        value: Felt,
//        _gas: &mut u128,
        _gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        println!("Called `storage_write({address_domain}, {address}, {value})` from MLIR.");
        println!("Called `storage_write({address_domain}, {address}, {value})` from MLIR.");
//        Ok(())
        Ok(())
//    }
    }
//

//    fn emit_event(&mut self, keys: &[Felt], data: &[Felt], _gas: &mut u128) -> SyscallResult<()> {
    fn emit_event(&mut self, keys: &[Felt], data: &[Felt], _gas: &mut u128) -> SyscallResult<()> {
//        println!("Called `emit_event({keys:?}, {data:?})` from MLIR.");
        println!("Called `emit_event({keys:?}, {data:?})` from MLIR.");
//        Ok(())
        Ok(())
//    }
    }
//

//    fn send_message_to_l1(
    fn send_message_to_l1(
//        &mut self,
        &mut self,
//        to_address: Felt,
        to_address: Felt,
//        payload: &[Felt],
        payload: &[Felt],
//        _gas: &mut u128,
        _gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        println!("Called `send_message_to_l1({to_address}, {payload:?})` from MLIR.");
        println!("Called `send_message_to_l1({to_address}, {payload:?})` from MLIR.");
//        Ok(())
        Ok(())
//    }
    }
//

//    fn keccak(
    fn keccak(
//        &mut self,
        &mut self,
//        input: &[u64],
        input: &[u64],
//        gas: &mut u128,
        gas: &mut u128,
//    ) -> SyscallResult<cairo_native::starknet::U256> {
    ) -> SyscallResult<cairo_native::starknet::U256> {
//        println!("Called `keccak({input:?})` from MLIR.");
        println!("Called `keccak({input:?})` from MLIR.");
//        *gas -= 1000;
        *gas -= 1000;
//        Ok(U256 {
        Ok(U256 {
//            hi: 0,
            hi: 0,
//            lo: 1234567890,
            lo: 1234567890,
//        })
        })
//    }
    }
//

//    fn secp256k1_new(
    fn secp256k1_new(
//        &mut self,
        &mut self,
//        _x: U256,
        _x: U256,
//        _y: U256,
        _y: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256k1Point>> {
    ) -> SyscallResult<Option<Secp256k1Point>> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256k1_add(
    fn secp256k1_add(
//        &mut self,
        &mut self,
//        _p0: Secp256k1Point,
        _p0: Secp256k1Point,
//        _p1: Secp256k1Point,
        _p1: Secp256k1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256k1Point> {
    ) -> SyscallResult<Secp256k1Point> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256k1_mul(
    fn secp256k1_mul(
//        &mut self,
        &mut self,
//        _p: Secp256k1Point,
        _p: Secp256k1Point,
//        _m: U256,
        _m: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256k1Point> {
    ) -> SyscallResult<Secp256k1Point> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256k1_get_point_from_x(
    fn secp256k1_get_point_from_x(
//        &mut self,
        &mut self,
//        _x: U256,
        _x: U256,
//        _y_parity: bool,
        _y_parity: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256k1Point>> {
    ) -> SyscallResult<Option<Secp256k1Point>> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256k1_get_xy(
    fn secp256k1_get_xy(
//        &mut self,
        &mut self,
//        _p: Secp256k1Point,
        _p: Secp256k1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(U256, U256)> {
    ) -> SyscallResult<(U256, U256)> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256r1_new(
    fn secp256r1_new(
//        &mut self,
        &mut self,
//        _x: U256,
        _x: U256,
//        _y: U256,
        _y: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256r1Point>> {
    ) -> SyscallResult<Option<Secp256r1Point>> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256r1_add(
    fn secp256r1_add(
//        &mut self,
        &mut self,
//        _p0: Secp256r1Point,
        _p0: Secp256r1Point,
//        _p1: Secp256r1Point,
        _p1: Secp256r1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256r1Point> {
    ) -> SyscallResult<Secp256r1Point> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256r1_mul(
    fn secp256r1_mul(
//        &mut self,
        &mut self,
//        _p: Secp256r1Point,
        _p: Secp256r1Point,
//        _m: U256,
        _m: U256,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256r1Point> {
    ) -> SyscallResult<Secp256r1Point> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256r1_get_point_from_x(
    fn secp256r1_get_point_from_x(
//        &mut self,
        &mut self,
//        _x: U256,
        _x: U256,
//        _y_parity: bool,
        _y_parity: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256r1Point>> {
    ) -> SyscallResult<Option<Secp256r1Point>> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn secp256r1_get_xy(
    fn secp256r1_get_xy(
//        &mut self,
        &mut self,
//        _p: Secp256r1Point,
        _p: Secp256r1Point,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(U256, U256)> {
    ) -> SyscallResult<(U256, U256)> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//}
}
//

//lazy_static! {
lazy_static! {
//    static ref KECCAK_CONTRACT: cairo_lang_starknet_classes::contract_class::ContractClass = {
    static ref KECCAK_CONTRACT: cairo_lang_starknet_classes::contract_class::ContractClass = {
//        let path = Path::new("tests/tests/starknet/contracts/test_keccak.cairo");
        let path = Path::new("tests/tests/starknet/contracts/test_keccak.cairo");
//

//        compile_path(
        compile_path(
//            path,
            path,
//            None,
            None,
//            CompilerConfig {
            CompilerConfig {
//                replace_ids: true,
                replace_ids: true,
//                ..Default::default()
                ..Default::default()
//            },
            },
//        )
        )
//        .unwrap()
        .unwrap()
//    };
    };
//}
}
//

//#[test]
#[test]
//fn keccak_test() {
fn keccak_test() {
//    let contract = &KECCAK_CONTRACT;
    let contract = &KECCAK_CONTRACT;
//

//    let entry_point = contract.entry_points_by_type.external.first().unwrap();
    let entry_point = contract.entry_points_by_type.external.first().unwrap();
//

//    let program = contract.extract_sierra_program().unwrap();
    let program = contract.extract_sierra_program().unwrap();
//    let result =
    let result =
//        run_native_starknet_contract(&program, entry_point.function_idx, &[], SyscallHandler);
        run_native_starknet_contract(&program, entry_point.function_idx, &[], SyscallHandler);
//

//    assert!(!result.failure_flag);
    assert!(!result.failure_flag);
//    assert_eq!(result.remaining_gas, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFEEDDB);
    assert_eq!(result.remaining_gas, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFEEDDB);
//    assert_eq!(result.return_values, vec![1.into()]);
    assert_eq!(result.return_values, vec![1.into()]);
//}
}
