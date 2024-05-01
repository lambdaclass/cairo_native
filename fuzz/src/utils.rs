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

pub fn setup_program(program_path: &str) -> NativeExecutor {
    let path = Path::new(program_path);

    let contract = compile_path(
        path,
        None,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )?;

    let entry_point = contract.entry_points_by_type.constructor.get(0)?;
    let sierra_program = contract.extract_sierra_program()?;
    let native_context = NativeContext::new();

    let mut native_program = native_context.compile(&sierra_program).unwrap();
    native_program
        .insert_metadata(SyscallHandlerMeta::new(&mut SyscallHandler))
        .unwrap();

    // Call the echo function from the contract using the generated wrapper.
    let entry_point_fn =
        find_entry_point_by_idx(&sierra_program, entry_point.function_idx).unwrap();

    let fn_id = &entry_point_fn.id;

    let native_executor = NativeExecutor::new(native_program);
}

#[derive(Debug)]
pub struct SyscallHandler;

impl StarknetSyscallHandler for SyscallHandler {
    fn get_block_hash(
        &mut self,
        block_number: u64,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Felt> {
        todo!()
    }

    fn get_execution_info(&mut self, remaining_gas: &mut u128) -> SyscallResult<ExecutionInfo> {
        todo!()
    }

    fn get_execution_info_v2(
        &mut self,
        remaining_gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::ExecutionInfoV2> {
        todo!()
    }

    fn deploy(
        &mut self,
        class_hash: Felt,
        contract_address_salt: Felt,
        calldata: &[Felt],
        deploy_from_zero: bool,
        remaining_gas: &mut u128,
    ) -> SyscallResult<(Felt, Vec<Felt>)> {
        todo!()
    }

    fn replace_class(&mut self, class_hash: Felt, remaining_gas: &mut u128) -> SyscallResult<()> {
        todo!()
    }

    fn library_call(
        &mut self,
        class_hash: Felt,
        function_selector: Felt,
        calldata: &[Felt],
        remaining_gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        todo!()
    }

    fn call_contract(
        &mut self,
        address: Felt,
        entry_point_selector: Felt,
        calldata: &[Felt],
        remaining_gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>> {
        todo!()
    }

    fn storage_read(
        &mut self,
        address_domain: u32,
        address: Felt,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Felt> {
        todo!()
    }

    fn storage_write(
        &mut self,
        address_domain: u32,
        address: Felt,
        value: Felt,
        remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        todo!()
    }

    fn emit_event(
        &mut self,
        keys: &[Felt],
        data: &[Felt],
        remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        todo!()
    }

    fn send_message_to_l1(
        &mut self,
        to_address: Felt,
        payload: &[Felt],
        remaining_gas: &mut u128,
    ) -> SyscallResult<()> {
        todo!()
    }

    fn keccak(&mut self, input: &[u64], remaining_gas: &mut u128) -> SyscallResult<U256> {
        todo!()
    }

    fn secp256k1_new(
        &mut self,
        x: U256,
        y: U256,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_add(
        &mut self,
        p0: cairo_native::starknet::Secp256k1Point,
        p1: cairo_native::starknet::Secp256k1Point,
        remaining_gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::Secp256k1Point> {
        todo!()
    }

    fn secp256k1_mul(
        &mut self,
        p: cairo_native::starknet::Secp256k1Point,
        m: U256,
        remaining_gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::Secp256k1Point> {
        todo!()
    }

    fn secp256k1_get_point_from_x(
        &mut self,
        x: U256,
        y_parity: bool,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256k1Point>> {
        todo!()
    }

    fn secp256k1_get_xy(
        &mut self,
        p: cairo_native::starknet::Secp256k1Point,
        remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)> {
        todo!()
    }

    fn secp256r1_new(
        &mut self,
        x: U256,
        y: U256,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256r1Point>> {
        todo!()
    }

    fn secp256r1_add(
        &mut self,
        p0: cairo_native::starknet::Secp256r1Point,
        p1: cairo_native::starknet::Secp256r1Point,
        remaining_gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::Secp256r1Point> {
        todo!()
    }

    fn secp256r1_mul(
        &mut self,
        p: cairo_native::starknet::Secp256r1Point,
        m: U256,
        remaining_gas: &mut u128,
    ) -> SyscallResult<cairo_native::starknet::Secp256r1Point> {
        todo!()
    }

    fn secp256r1_get_point_from_x(
        &mut self,
        x: U256,
        y_parity: bool,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<cairo_native::starknet::Secp256r1Point>> {
        todo!()
    }

    fn secp256r1_get_xy(
        &mut self,
        p: cairo_native::starknet::Secp256r1Point,
        remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)> {
        todo!()
    }
}
