#![feature(strict_provenance)]

use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program_registry::ProgramRegistry,
};
use cairo_native::{
    metadata::{
        runtime_bindings::RuntimeBindingsMeta, syscall_handler::SyscallHandlerMeta, MetadataStorage,
    },
    starknet::{StarkNetSyscallHandler, SyscallResult},
};
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_passes},
    Context, ExecutionEngine,
};
use serde_json::json;
use std::{io, path::Path};

#[derive(Debug)]
struct SyscallHandler;

impl StarkNetSyscallHandler for SyscallHandler {
    fn get_block_hash(&self, block_number: u64) -> SyscallResult<cairo_felt::Felt252> {
        println!("Called `get_block_hash({block_number})` from MLIR.");
        Ok(todo!())
    }

    fn get_execution_info(&self) -> SyscallResult<cairo_native::starknet::ExecutionInfo> {
        todo!()
    }

    fn deploy(
        &self,
        _class_hash: cairo_felt::Felt252,
        _contract_address_salt: cairo_felt::Felt252,
        _calldata: &[cairo_felt::Felt252],
        _deploy_from_zero: bool,
    ) -> SyscallResult<(cairo_felt::Felt252, Vec<cairo_felt::Felt252>)> {
        todo!()
    }

    fn replace_class(&self, _class_hash: cairo_felt::Felt252) -> SyscallResult<()> {
        todo!()
    }

    fn library_call(
        &self,
        _class_hash: cairo_felt::Felt252,
        _function_selector: cairo_felt::Felt252,
        _calldata: &[cairo_felt::Felt252],
    ) -> SyscallResult<Vec<cairo_felt::Felt252>> {
        todo!()
    }

    fn call_contract(
        &self,
        _address: cairo_felt::Felt252,
        _entry_point_selector: cairo_felt::Felt252,
        _calldata: &[cairo_felt::Felt252],
    ) -> SyscallResult<Vec<cairo_felt::Felt252>> {
        todo!()
    }

    fn storage_read(
        &self,
        _address_domain: u32,
        _address: cairo_felt::Felt252,
    ) -> SyscallResult<cairo_felt::Felt252> {
        todo!()
    }

    fn storage_write(
        &self,
        _address_domain: u32,
        _address: cairo_felt::Felt252,
        _value: cairo_felt::Felt252,
    ) -> SyscallResult<()> {
        todo!()
    }

    fn emit_event(
        &self,
        _keys: &[cairo_felt::Felt252],
        _data: &[cairo_felt::Felt252],
    ) -> SyscallResult<()> {
        todo!()
    }

    fn send_message_to_l1(
        &self,
        _to_address: cairo_felt::Felt252,
        _payload: &[cairo_felt::Felt252],
    ) -> SyscallResult<()> {
        todo!()
    }

    fn keccak(&self, _input: &[u64]) -> SyscallResult<cairo_native::starknet::U256> {
        todo!()
    }

    fn secp256k1_add(
        &self,
        _p0: cairo_native::starknet::Secp256k1Point,
        _p1: cairo_native::starknet::Secp256k1Point,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_get_point_from_x(
        &self,
        _x: cairo_native::starknet::U256,
        _y_parity: bool,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_get_xy(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
    ) -> SyscallResult<(cairo_native::starknet::U256, cairo_native::starknet::U256)> {
        todo!()
    }

    fn secp256k1_mul(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
        _m: cairo_native::starknet::U256,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_new(
        &self,
        _x: cairo_native::starknet::U256,
        _y: cairo_native::starknet::U256,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_add(
        &self,
        _p0: cairo_native::starknet::Secp256k1Point,
        _p1: cairo_native::starknet::Secp256k1Point,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_get_point_from_x(
        &self,
        _x: cairo_native::starknet::U256,
        _y_parity: bool,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_get_xy(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
    ) -> SyscallResult<(cairo_native::starknet::U256, cairo_native::starknet::U256)> {
        todo!()
    }

    fn secp256r1_mul(
        &self,
        _p: cairo_native::starknet::Secp256k1Point,
        _m: cairo_native::starknet::U256,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256r1_new(
        &self,
        _x: cairo_native::starknet::U256,
        _y: cairo_native::starknet::U256,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn pop_log(&self) {
        todo!()
    }

    fn set_account_contract_address(&self, _contract_address: cairo_felt::Felt252) {
        todo!()
    }

    fn set_block_number(&self, _block_number: u64) {
        todo!()
    }

    fn set_block_timestamp(&self, _block_timestamp: u64) {
        todo!()
    }

    fn set_caller_address(&self, _address: cairo_felt::Felt252) {
        todo!()
    }

    fn set_chain_id(&self, _chain_id: cairo_felt::Felt252) {
        todo!()
    }

    fn set_contract_address(&self, _address: cairo_felt::Felt252) {
        todo!()
    }

    fn set_max_fee(&self, _max_fee: u128) {
        todo!()
    }

    fn set_nonce(&self, _nonce: cairo_felt::Felt252) {
        todo!()
    }

    fn set_sequencer_address(&self, _address: cairo_felt::Felt252) {
        todo!()
    }

    fn set_signature(&self, _signature: &[cairo_felt::Felt252]) {
        todo!()
    }

    fn set_transaction_hash(&self, _transaction_hash: cairo_felt::Felt252) {
        todo!()
    }

    fn set_version(&self, _version: cairo_felt::Felt252) {
        todo!()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // FIXME: Remove when cairo adds an easy to use API for setting the corelibs path.
    std::env::set_var(
        "CARGO_MANIFEST_DIR",
        format!("{}/a", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
    );

    let program = cairo_lang_compiler::compile_cairo_project_at_path(
        Path::new("programs/examples/hello_starknet.cairo"),
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let entry_point = match program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some("hello_starknet::hello_starknet::main"))
    {
        Some(x) => x,
        None => {
            // TODO: Use a proper error.
            eprintln!("Entry point `hello_starknet::hello_starknet::main` not found in program.");
            return Ok(());
        }
    };

    // Initialize MLIR.
    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();

    register_all_passes();

    // Compile the program.
    let mut module = Module::new(Location::unknown(&context));
    let mut metadata = MetadataStorage::new();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program)?;

    // Make the runtime library available.
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    // Make the Starknet syscall handler available.
    metadata
        .insert(SyscallHandlerMeta::new(&SyscallHandler))
        .unwrap();

    cairo_native::compile::<CoreType, CoreLibfunc>(
        &context,
        &module,
        &program,
        &registry,
        &mut metadata,
        None,
    )?;

    // Lower to LLVM.
    let pass_manager = PassManager::new(&context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());

    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());

    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_index_to_llvm_pass());
    pass_manager.add_pass(pass::conversion::create_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

    pass_manager.run(&mut module)?;

    // Create the JIT engine.
    let engine = ExecutionEngine::new(&module, 3, &[], false);

    #[cfg(feature = "with-runtime")]
    unsafe {
        engine.register_symbol(
            "cairo_native__libfunc__debug__print",
            cairo_native_runtime::cairo_native__libfunc__debug__print
                as *const fn(i32, *const [u8; 32], usize) -> i32 as *mut (),
        );

        engine.register_symbol(
            "cairo_native__libfunc_pedersen",
            cairo_native_runtime::cairo_native__libfunc_pedersen
                as *const fn(*mut u8, *mut u8, *mut u8) -> () as *mut (),
        );
    }

    let params_input = json!([
        u64::MAX,
        metadata
            .get::<SyscallHandlerMeta>()
            .unwrap()
            .as_ptr()
            .addr()
    ]);

    cairo_native::execute(
        &engine,
        &registry,
        &entry_point.id,
        params_input,
        &mut serde_json::Serializer::pretty(io::stdout()),
    )
    .unwrap();

    Ok(())
}
