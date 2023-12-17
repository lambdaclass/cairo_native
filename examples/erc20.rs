use cairo_lang_compiler::CompilerConfig;
use cairo_lang_starknet::contract_class::compile_path;
use cairo_native::context::NativeContext;
use cairo_native::executor::JitNativeExecutor;
use cairo_native::utils::find_entry_point_by_idx;
use cairo_native::{
    metadata::syscall_handler::SyscallHandlerMeta,
    starknet::{BlockInfo, ExecutionInfo, StarkNetSyscallHandler, SyscallResult, TxInfo, U256},
};
use starknet_types_core::felt::Felt;
use std::path::Path;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Debug)]
struct SyscallHandler;

impl StarkNetSyscallHandler for SyscallHandler {
    fn get_block_hash(&mut self, block_number: u64, _gas: &mut u128) -> SyscallResult<Felt> {
        println!("Called `get_block_hash({block_number})` from MLIR.");
        Ok(Felt::from_bytes_be_slice(b"get_block_hash ok"))
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
        class_hash: Felt,
        contract_address_salt: Felt,
        calldata: &[Felt],
        deploy_from_zero: bool,
        _gas: &mut u128,
    ) -> SyscallResult<(Felt, Vec<Felt>)> {
        println!("Called `deploy({class_hash}, {contract_address_salt}, {calldata:?}, {deploy_from_zero})` from MLIR.");
        Ok((
            class_hash + contract_address_salt,
            calldata.iter().map(|x| x + Felt::from(1)).collect(),
        ))
    }

    fn replace_class(&mut self, class_hash: Felt, _gas: &mut u128) -> SyscallResult<()> {
        println!("Called `replace_class({class_hash})` from MLIR.");
        Ok(())
    }

    fn library_call(
        &mut self,
        class_hash: Felt,
        function_selector: Felt,
        calldata: &[Felt],
        _gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        println!(
            "Called `library_call({class_hash}, {function_selector}, {calldata:?})` from MLIR."
        );
        Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
    }

    fn call_contract(
        &mut self,
        address: Felt,
        entry_point_selector: Felt,
        calldata: &[Felt],
        _gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        println!(
            "Called `call_contract({address}, {entry_point_selector}, {calldata:?})` from MLIR."
        );
        Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
    }

    fn storage_read(
        &mut self,
        address_domain: u32,
        address: Felt,
        _gas: &mut u128,
    ) -> SyscallResult<Felt> {
        println!("Called `storage_read({address_domain}, {address})` from MLIR.");
        Ok(address * Felt::from(3))
    }

    fn storage_write(
        &mut self,
        address_domain: u32,
        address: Felt,
        value: Felt,
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `storage_write({address_domain}, {address}, {value})` from MLIR.");
        Ok(())
    }

    fn emit_event(&mut self, keys: &[Felt], data: &[Felt], _gas: &mut u128) -> SyscallResult<()> {
        println!("Called `emit_event({keys:?}, {data:?})` from MLIR.");
        Ok(())
    }

    fn send_message_to_l1(
        &mut self,
        to_address: Felt,
        payload: &[Felt],
        _gas: &mut u128,
    ) -> SyscallResult<()> {
        println!("Called `send_message_to_l1({to_address}, {payload:?})` from MLIR.");
        Ok(())
    }

    fn keccak(
        &mut self,
        input: &[u64],
        _gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::U256> {
        println!("Called `keccak({input:?})` from MLIR.");
        Ok(U256(Felt::from(1234567890).to_bytes_le()))
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

    fn set_account_contract_address(&mut self, _contract_address: Felt) {
        todo!()
    }

    fn set_block_number(&mut self, _block_number: u64) {
        todo!()
    }

    fn set_block_timestamp(&mut self, _block_timestamp: u64) {
        todo!()
    }

    fn set_caller_address(&mut self, _address: Felt) {
        todo!()
    }

    fn set_chain_id(&mut self, _chain_id: Felt) {
        todo!()
    }

    fn set_contract_address(&mut self, _address: Felt) {
        todo!()
    }

    fn set_max_fee(&mut self, _max_fee: u128) {
        todo!()
    }

    fn set_nonce(&mut self, _nonce: Felt) {
        todo!()
    }

    fn set_sequencer_address(&mut self, _address: Felt) {
        todo!()
    }

    fn set_signature(&mut self, _signature: &[Felt]) {
        todo!()
    }

    fn set_transaction_hash(&mut self, _transaction_hash: Felt) {
        todo!()
    }

    fn set_version(&mut self, _version: Felt) {
        todo!()
    }
}

fn main() {
    #[cfg(not(feature = "with-runtime"))]
    compile_error!("This example requires the `with-runtime` feature to be active.");

    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )
    .unwrap();

    let path = Path::new("programs/erc20.cairo");

    let contract = compile_path(
        path,
        None,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let entry_point = contract.entry_points_by_type.constructor.get(0).unwrap();
    let sierra_program = contract.extract_sierra_program().unwrap();

    let native_context = NativeContext::new();

    let mut native_program = native_context.compile(&sierra_program).unwrap();
    native_program
        .insert_metadata(SyscallHandlerMeta::new(&mut SyscallHandler))
        .unwrap();

    let entry_point_fn =
        find_entry_point_by_idx(&sierra_program, entry_point.function_idx).unwrap();
    let fn_id = &entry_point_fn.id;

    let native_executor = JitNativeExecutor::new(native_program);

    let result = native_executor
        .execute_contract(
            fn_id,
            &[
                Felt::from_bytes_be_slice(b"name"),
                Felt::from_bytes_be_slice(b"symbol"),
                Felt::from(0),
                Felt::from(i64::MAX),
                Felt::from(4),
                Felt::from(6),
            ],
            Some(u128::MAX),
        )
        .expect("failed to execute the given contract");

    println!();
    println!("Cairo program was compiled and executed successfully.");
    println!("{result:#?}");
}
