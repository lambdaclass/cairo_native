use cairo_felt::Felt252;
use cairo_lang_compiler::CompilerConfig;
use cairo_lang_starknet::contract_class::compile_path;
use cairo_native::context::NativeContext;
use cairo_native::executor::NativeExecutor;
use cairo_native::{
    metadata::syscall_handler::SyscallHandlerMeta,
    starknet::{BlockInfo, ExecutionInfo, StarkNetSyscallHandler, SyscallResult, TxInfo, U256},
    utils::{felt252_bigint, felt252_short_str, find_function_id},
};
use serde_json::json;
use std::io;
use std::path::Path;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Debug)]
struct SyscallHandler;

impl StarkNetSyscallHandler for SyscallHandler {
    fn get_block_hash(
        &mut self,
        block_number: u64,
        _gas: &mut u64,
    ) -> SyscallResult<cairo_felt::Felt252> {
        println!("Called `get_block_hash({block_number})` from MLIR.");
        Ok(Felt252::from_bytes_be(b"get_block_hash ok"))
    }

    fn get_execution_info(
        &self,
        _gas: &mut u64,
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
        class_hash: cairo_felt::Felt252,
        contract_address_salt: cairo_felt::Felt252,
        calldata: &[cairo_felt::Felt252],
        deploy_from_zero: bool,
        _gas: &mut u64,
    ) -> SyscallResult<(cairo_felt::Felt252, Vec<cairo_felt::Felt252>)> {
        println!("Called `deploy({class_hash}, {contract_address_salt}, {calldata:?}, {deploy_from_zero})` from MLIR.");
        Ok((
            class_hash + contract_address_salt,
            calldata.iter().map(|x| x + &Felt252::new(1)).collect(),
        ))
    }

    fn replace_class(&mut self, class_hash: cairo_felt::Felt252, _gas: &mut u64) -> SyscallResult<()> {
        println!("Called `replace_class({class_hash})` from MLIR.");
        Ok(())
    }

    fn library_call(
        &mut self,
        class_hash: cairo_felt::Felt252,
        function_selector: cairo_felt::Felt252,
        calldata: &[cairo_felt::Felt252],
        _gas: &mut u64,
    ) -> SyscallResult<Vec<cairo_felt::Felt252>> {
        println!(
            "Called `library_call({class_hash}, {function_selector}, {calldata:?})` from MLIR."
        );
        Ok(calldata.iter().map(|x| x * &Felt252::new(3)).collect())
    }

    fn call_contract(
        &mut self,
        address: cairo_felt::Felt252,
        entry_point_selector: cairo_felt::Felt252,
        calldata: &[cairo_felt::Felt252],
        _gas: &mut u64,
    ) -> SyscallResult<Vec<cairo_felt::Felt252>> {
        println!(
            "Called `call_contract({address}, {entry_point_selector}, {calldata:?})` from MLIR."
        );
        Ok(calldata.iter().map(|x| x * &Felt252::new(3)).collect())
    }

    fn storage_read(
        &mut self,
        address_domain: u32,
        address: cairo_felt::Felt252,
        _gas: &mut u64,
    ) -> SyscallResult<cairo_felt::Felt252> {
        println!("Called `storage_read({address_domain}, {address})` from MLIR.");
        Ok(address * &Felt252::new(3))
    }

    fn storage_write(
        &mut self,
        address_domain: u32,
        address: cairo_felt::Felt252,
        value: cairo_felt::Felt252,
        _gas: &mut u64,
    ) -> SyscallResult<()> {
        println!("Called `storage_write({address_domain}, {address}, {value})` from MLIR.");
        Ok(())
    }

    fn emit_event(
        &mut self,
        keys: &[cairo_felt::Felt252],
        data: &[cairo_felt::Felt252],
        _gas: &mut u64,
    ) -> SyscallResult<()> {
        println!("Called `emit_event({keys:?}, {data:?})` from MLIR.");
        Ok(())
    }

    fn send_message_to_l1(
        &mut self,
        to_address: cairo_felt::Felt252,
        payload: &[cairo_felt::Felt252],
        _gas: &mut u64,
    ) -> SyscallResult<()> {
        println!("Called `send_message_to_l1({to_address}, {payload:?})` from MLIR.");
        Ok(())
    }

    fn keccak(&self, input: &[u64], _gas: &mut u64) -> SyscallResult<cairo_native::starknet::U256> {
        println!("Called `keccak({input:?})` from MLIR.");
        Ok(U256(Felt252::from(1234567890).to_le_bytes()))
    }

    fn secp256k1_add(
        &mut self,
        _p0: cairo_native::starknet::Secp256k1Point,
        _p1: cairo_native::starknet::Secp256k1Point,
        _gas: &mut u64,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_get_point_from_x(
        &self,
        _x: cairo_native::starknet::U256,
        _y_parity: bool,
        _gas: &mut u64,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_get_xy(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
        _gas: &mut u64,
    ) -> SyscallResult<(cairo_native::starknet::U256, cairo_native::starknet::U256)> {
        todo!()
    }

    fn secp256k1_mul(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
        _m: cairo_native::starknet::U256,
        _gas: &mut u64,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_new(
        &self,
        _x: cairo_native::starknet::U256,
        _y: cairo_native::starknet::U256,
        _gas: &mut u64,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_add(
        &self,
        _p0: cairo_native::starknet::Secp256k1Point,
        _p1: cairo_native::starknet::Secp256k1Point,
        _gas: &mut u64,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_get_point_from_x(
        &self,
        _x: cairo_native::starknet::U256,
        _y_parity: bool,
        _gas: &mut u64,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_get_xy(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
        _gas: &mut u64,
    ) -> SyscallResult<(cairo_native::starknet::U256, cairo_native::starknet::U256)> {
        todo!()
    }

    fn secp256r1_mul(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
        _m: cairo_native::starknet::U256,
        _gas: &mut u64,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_new(
        &mut self,
        _x: cairo_native::starknet::U256,
        _y: cairo_native::starknet::U256,
        _gas: &mut u64,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn pop_log(&mut self) {
        todo!()
    }

    fn set_account_contract_address(&mut self, _contract_address: cairo_felt::Felt252) {
        todo!()
    }

    fn set_block_number(&mut self, _block_number: u64) {
        todo!()
    }

    fn set_block_timestamp(&mut self, _block_timestamp: u64) {
        todo!()
    }

    fn set_caller_address(&mut self, _address: cairo_felt::Felt252) {
        todo!()
    }

    fn set_chain_id(&mut self, _chain_id: cairo_felt::Felt252) {
        todo!()
    }

    fn set_contract_address(&mut self, _address: cairo_felt::Felt252) {
        todo!()
    }

    fn set_max_fee(&mut self, _max_fee: u128) {
        todo!()
    }

    fn set_nonce(&mut self, _nonce: cairo_felt::Felt252) {
        todo!()
    }

    fn set_sequencer_address(&mut self, _address: cairo_felt::Felt252) {
        todo!()
    }

    fn set_signature(&mut self, _signature: &[cairo_felt::Felt252]) {
        todo!()
    }

    fn set_transaction_hash(&mut self, _transaction_hash: cairo_felt::Felt252) {
        todo!()
    }

    fn set_version(&mut self, _version: cairo_felt::Felt252) {
        todo!()
    }
}

fn main() {
    // FIXME: Remove when cairo adds an easy to use API for setting the corelibs path.
    std::env::set_var(
        "CARGO_MANIFEST_DIR",
        format!("{}/a", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
    );

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

    let sierra_program = contract.extract_sierra_program().unwrap();

    // uncomment to save the contract sierra program
    // std::fs::write("erc20.sierra", sierra_program.to_string()).unwrap();

    /* uncomment to find all the functions in the program you can call
    let names: Vec<_> = sierra_program
        .funcs
        .iter()
        .map(|x| x.id.debug_name.as_ref())
        .collect();
    println!("{names:#?}");
    */

    let native_context = NativeContext::new();

    let mut native_program = native_context.compile(&sierra_program).unwrap();
    native_program
        .insert_metadata(SyscallHandlerMeta::new(&SyscallHandler))
        .unwrap();

    let syscall_addr = native_program
        .get_metadata::<SyscallHandlerMeta>()
        .unwrap()
        .as_ptr()
        .as_ptr() as *const () as usize;

    let fn_id = find_function_id(
        &sierra_program,
        "erc20::erc20::erc_20::__wrapper_constructor",
    );
    let required_init_gas = native_program.get_required_init_gas(fn_id);

    let params = json!([
        // pedersen
        null,
        // range check
        null,
        // gas
        u64::MAX,
        // system
        syscall_addr,
        // The amount of params change depending on the contract function called
        // Struct<Span<Array<felt>>>
        [
            // Span<Array<felt>>
            [
                // contract state

                // name
                felt252_short_str("name"),   // name
                felt252_short_str("symbol"), // symbol
                felt252_bigint(0),           // decimals
                felt252_bigint(i64::MAX),    // initial supply
                felt252_bigint(4),           // contract address
                felt252_bigint(6),           // ??
            ]
        ]
    ]);

    let returns = &mut serde_json::Serializer::pretty(io::stdout());

    let native_executor = NativeExecutor::new(native_program);

    native_executor
        .execute(fn_id, params, returns, required_init_gas)
        .unwrap();

    // Useful code to print a short felt string returned from the json, in case it panics
    /*
    let data = vec![
        1919251315, 543253604, 1751457840, 1953439860, 1768846368, 809115757, 1163019058, 0,
    ];

    let value = Felt252::new(BigUint::new(data));

    if let Some(shortstring) = as_cairo_short_string(&value) {
        println!("[DEBUG]\t{shortstring: <31}\t(raw: {})", value.to_bigint())
    }
    */

    println!("Cairo program was compiled and executed succesfully.");
}
