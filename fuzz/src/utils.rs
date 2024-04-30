use cairo_native::starknet::{StarknetSyscallHandler, SyscallResult, U256, ExecutionInfo};
use starknet_types_core::felt::Felt;


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

    fn get_execution_info_v2(&mut self, remaining_gas: &mut u128)
        -> SyscallResult<cairo_native::starknet::ExecutionInfoV2> {
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
