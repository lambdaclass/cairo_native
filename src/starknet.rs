////! Starknet related code for `cairo_native`
//! Starknet related code for `cairo_native`
//

//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//

//pub type SyscallResult<T> = std::result::Result<T, Vec<Felt>>;
pub type SyscallResult<T> = std::result::Result<T, Vec<Felt>>;
//

//#[repr(C)]
#[repr(C)]
//#[derive(Debug)]
#[derive(Debug)]
//pub struct ArrayAbi<T> {
pub struct ArrayAbi<T> {
//    pub ptr: *mut T,
    pub ptr: *mut T,
//    pub since: u32,
    pub since: u32,
//    pub until: u32,
    pub until: u32,
//    pub capacity: u32,
    pub capacity: u32,
//}
}
//

///// Binary representation of a `Felt` (in MLIR).
/// Binary representation of a `Felt` (in MLIR).
//#[derive(Debug, Clone)]
#[derive(Debug, Clone)]
//#[repr(C, align(16))]
#[repr(C, align(16))]
//pub struct Felt252Abi(pub [u8; 32]);
pub struct Felt252Abi(pub [u8; 32]);
///// Binary representation of a `u256` (in MLIR).
/// Binary representation of a `u256` (in MLIR).
//// TODO: This shouldn't need to be public.
// TODO: This shouldn't need to be public.
//#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//#[repr(C, align(16))]
#[repr(C, align(16))]
//pub struct U256 {
pub struct U256 {
//    pub hi: u128,
    pub hi: u128,
//    pub lo: u128,
    pub lo: u128,
//}
}
//

//#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//pub struct ExecutionInfo {
pub struct ExecutionInfo {
//    pub block_info: BlockInfo,
    pub block_info: BlockInfo,
//    pub tx_info: TxInfo,
    pub tx_info: TxInfo,
//    pub caller_address: Felt,
    pub caller_address: Felt,
//    pub contract_address: Felt,
    pub contract_address: Felt,
//    pub entry_point_selector: Felt,
    pub entry_point_selector: Felt,
//}
}
//

//#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//pub struct ExecutionInfoV2 {
pub struct ExecutionInfoV2 {
//    pub block_info: BlockInfo,
    pub block_info: BlockInfo,
//    pub tx_info: TxV2Info,
    pub tx_info: TxV2Info,
//    pub caller_address: Felt,
    pub caller_address: Felt,
//    pub contract_address: Felt,
    pub contract_address: Felt,
//    pub entry_point_selector: Felt,
    pub entry_point_selector: Felt,
//}
}
//

//#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//pub struct TxV2Info {
pub struct TxV2Info {
//    pub version: Felt,
    pub version: Felt,
//    pub account_contract_address: Felt,
    pub account_contract_address: Felt,
//    pub max_fee: u128,
    pub max_fee: u128,
//    pub signature: Vec<Felt>,
    pub signature: Vec<Felt>,
//    pub transaction_hash: Felt,
    pub transaction_hash: Felt,
//    pub chain_id: Felt,
    pub chain_id: Felt,
//    pub nonce: Felt,
    pub nonce: Felt,
//    pub resource_bounds: Vec<ResourceBounds>,
    pub resource_bounds: Vec<ResourceBounds>,
//    pub tip: u128,
    pub tip: u128,
//    pub paymaster_data: Vec<Felt>,
    pub paymaster_data: Vec<Felt>,
//    pub nonce_data_availability_mode: u32,
    pub nonce_data_availability_mode: u32,
//    pub fee_data_availability_mode: u32,
    pub fee_data_availability_mode: u32,
//    pub account_deployment_data: Vec<Felt>,
    pub account_deployment_data: Vec<Felt>,
//}
}
//

//#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//pub struct ResourceBounds {
pub struct ResourceBounds {
//    pub resource: Felt,
    pub resource: Felt,
//    pub max_amount: u64,
    pub max_amount: u64,
//    pub max_price_per_unit: u128,
    pub max_price_per_unit: u128,
//}
}
//

//#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//pub struct BlockInfo {
pub struct BlockInfo {
//    pub block_number: u64,
    pub block_number: u64,
//    pub block_timestamp: u64,
    pub block_timestamp: u64,
//    pub sequencer_address: Felt,
    pub sequencer_address: Felt,
//}
}
//

//#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
//#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "with-serde", derive(serde::Serialize, serde::Deserialize))]
//pub struct TxInfo {
pub struct TxInfo {
//    pub version: Felt,
    pub version: Felt,
//    pub account_contract_address: Felt,
    pub account_contract_address: Felt,
//    pub max_fee: u128,
    pub max_fee: u128,
//    pub signature: Vec<Felt>,
    pub signature: Vec<Felt>,
//    pub transaction_hash: Felt,
    pub transaction_hash: Felt,
//    pub chain_id: Felt,
    pub chain_id: Felt,
//    pub nonce: Felt,
    pub nonce: Felt,
//}
}
//

//#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
//pub struct Secp256k1Point {
pub struct Secp256k1Point {
//    pub x: U256,
    pub x: U256,
//    pub y: U256,
    pub y: U256,
//}
}
//

//#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
//pub struct Secp256r1Point {
pub struct Secp256r1Point {
//    pub x: U256,
    pub x: U256,
//    pub y: U256,
    pub y: U256,
//}
}
//

//pub trait StarknetSyscallHandler {
pub trait StarknetSyscallHandler {
//    fn get_block_hash(
    fn get_block_hash(
//        &mut self,
        &mut self,
//        block_number: u64,
        block_number: u64,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Felt>;
    ) -> SyscallResult<Felt>;
//

//    fn get_execution_info(&mut self, remaining_gas: &mut u128) -> SyscallResult<ExecutionInfo>;
    fn get_execution_info(&mut self, remaining_gas: &mut u128) -> SyscallResult<ExecutionInfo>;
//

//    fn get_execution_info_v2(&mut self, remaining_gas: &mut u128)
    fn get_execution_info_v2(&mut self, remaining_gas: &mut u128)
//        -> SyscallResult<ExecutionInfoV2>;
        -> SyscallResult<ExecutionInfoV2>;
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
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<(Felt, Vec<Felt>)>;
    ) -> SyscallResult<(Felt, Vec<Felt>)>;
//    fn replace_class(&mut self, class_hash: Felt, remaining_gas: &mut u128) -> SyscallResult<()>;
    fn replace_class(&mut self, class_hash: Felt, remaining_gas: &mut u128) -> SyscallResult<()>;
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
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Vec<Felt>>;
    ) -> SyscallResult<Vec<Felt>>;
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
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Vec<Felt>>;
    ) -> SyscallResult<Vec<Felt>>;
//

//    fn storage_read(
    fn storage_read(
//        &mut self,
        &mut self,
//        address_domain: u32,
        address_domain: u32,
//        address: Felt,
        address: Felt,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Felt>;
    ) -> SyscallResult<Felt>;
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
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<()>;
    ) -> SyscallResult<()>;
//

//    fn emit_event(
    fn emit_event(
//        &mut self,
        &mut self,
//        keys: &[Felt],
        keys: &[Felt],
//        data: &[Felt],
        data: &[Felt],
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<()>;
    ) -> SyscallResult<()>;
//

//    fn send_message_to_l1(
    fn send_message_to_l1(
//        &mut self,
        &mut self,
//        to_address: Felt,
        to_address: Felt,
//        payload: &[Felt],
        payload: &[Felt],
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<()>;
    ) -> SyscallResult<()>;
//

//    fn keccak(&mut self, input: &[u64], remaining_gas: &mut u128) -> SyscallResult<U256>;
    fn keccak(&mut self, input: &[u64], remaining_gas: &mut u128) -> SyscallResult<U256>;
//

//    fn secp256k1_new(
    fn secp256k1_new(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y: U256,
        y: U256,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256k1Point>>;
    ) -> SyscallResult<Option<Secp256k1Point>>;
//

//    fn secp256k1_add(
    fn secp256k1_add(
//        &mut self,
        &mut self,
//        p0: Secp256k1Point,
        p0: Secp256k1Point,
//        p1: Secp256k1Point,
        p1: Secp256k1Point,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256k1Point>;
    ) -> SyscallResult<Secp256k1Point>;
//

//    fn secp256k1_mul(
    fn secp256k1_mul(
//        &mut self,
        &mut self,
//        p: Secp256k1Point,
        p: Secp256k1Point,
//        m: U256,
        m: U256,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256k1Point>;
    ) -> SyscallResult<Secp256k1Point>;
//

//    fn secp256k1_get_point_from_x(
    fn secp256k1_get_point_from_x(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y_parity: bool,
        y_parity: bool,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256k1Point>>;
    ) -> SyscallResult<Option<Secp256k1Point>>;
//

//    fn secp256k1_get_xy(
    fn secp256k1_get_xy(
//        &mut self,
        &mut self,
//        p: Secp256k1Point,
        p: Secp256k1Point,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<(U256, U256)>;
    ) -> SyscallResult<(U256, U256)>;
//

//    fn secp256r1_new(
    fn secp256r1_new(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y: U256,
        y: U256,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256r1Point>>;
    ) -> SyscallResult<Option<Secp256r1Point>>;
//

//    fn secp256r1_add(
    fn secp256r1_add(
//        &mut self,
        &mut self,
//        p0: Secp256r1Point,
        p0: Secp256r1Point,
//        p1: Secp256r1Point,
        p1: Secp256r1Point,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256r1Point>;
    ) -> SyscallResult<Secp256r1Point>;
//

//    fn secp256r1_mul(
    fn secp256r1_mul(
//        &mut self,
        &mut self,
//        p: Secp256r1Point,
        p: Secp256r1Point,
//        m: U256,
        m: U256,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Secp256r1Point>;
    ) -> SyscallResult<Secp256r1Point>;
//

//    fn secp256r1_get_point_from_x(
    fn secp256r1_get_point_from_x(
//        &mut self,
        &mut self,
//        x: U256,
        x: U256,
//        y_parity: bool,
        y_parity: bool,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<Option<Secp256r1Point>>;
    ) -> SyscallResult<Option<Secp256r1Point>>;
//

//    fn secp256r1_get_xy(
    fn secp256r1_get_xy(
//        &mut self,
        &mut self,
//        p: Secp256r1Point,
        p: Secp256r1Point,
//        remaining_gas: &mut u128,
        remaining_gas: &mut u128,
//    ) -> SyscallResult<(U256, U256)>;
    ) -> SyscallResult<(U256, U256)>;
//

//    // Testing syscalls.
    // Testing syscalls.
//    fn pop_log(&mut self) {
    fn pop_log(&mut self) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_account_contract_address(&mut self, _contract_address: Felt) {
    fn set_account_contract_address(&mut self, _contract_address: Felt) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_block_number(&mut self, _block_number: u64) {
    fn set_block_number(&mut self, _block_number: u64) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_block_timestamp(&mut self, _block_timestamp: u64) {
    fn set_block_timestamp(&mut self, _block_timestamp: u64) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_caller_address(&mut self, _address: Felt) {
    fn set_caller_address(&mut self, _address: Felt) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_chain_id(&mut self, _chain_id: Felt) {
    fn set_chain_id(&mut self, _chain_id: Felt) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_contract_address(&mut self, _address: Felt) {
    fn set_contract_address(&mut self, _address: Felt) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_max_fee(&mut self, _max_fee: u128) {
    fn set_max_fee(&mut self, _max_fee: u128) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_nonce(&mut self, _nonce: Felt) {
    fn set_nonce(&mut self, _nonce: Felt) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_sequencer_address(&mut self, _address: Felt) {
    fn set_sequencer_address(&mut self, _address: Felt) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_signature(&mut self, _signature: &[Felt]) {
    fn set_signature(&mut self, _signature: &[Felt]) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_transaction_hash(&mut self, _transaction_hash: Felt) {
    fn set_transaction_hash(&mut self, _transaction_hash: Felt) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn set_version(&mut self, _version: Felt) {
    fn set_version(&mut self, _version: Felt) {
//        unimplemented!()
        unimplemented!()
//    }
    }
//}
}
//

//pub struct DummySyscallHandler;
pub struct DummySyscallHandler;
//

//impl StarknetSyscallHandler for DummySyscallHandler {
impl StarknetSyscallHandler for DummySyscallHandler {
//    fn get_block_hash(
    fn get_block_hash(
//        &mut self,
        &mut self,
//        _block_number: u64,
        _block_number: u64,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Felt> {
    ) -> SyscallResult<Felt> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn get_execution_info(&mut self, _remaining_gas: &mut u128) -> SyscallResult<ExecutionInfo> {
    fn get_execution_info(&mut self, _remaining_gas: &mut u128) -> SyscallResult<ExecutionInfo> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn get_execution_info_v2(
    fn get_execution_info_v2(
//        &mut self,
        &mut self,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<ExecutionInfoV2> {
    ) -> SyscallResult<ExecutionInfoV2> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn deploy(
    fn deploy(
//        &mut self,
        &mut self,
//        _class_hash: Felt,
        _class_hash: Felt,
//        _contract_address_salt: Felt,
        _contract_address_salt: Felt,
//        _calldata: &[Felt],
        _calldata: &[Felt],
//        _deploy_from_zero: bool,
        _deploy_from_zero: bool,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<(Felt, Vec<Felt>)> {
    ) -> SyscallResult<(Felt, Vec<Felt>)> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
    fn replace_class(&mut self, _class_hash: Felt, _remaining_gas: &mut u128) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn library_call(
    fn library_call(
//        &mut self,
        &mut self,
//        _class_hash: Felt,
        _class_hash: Felt,
//        _function_selector: Felt,
        _function_selector: Felt,
//        _calldata: &[Felt],
        _calldata: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Vec<Felt>> {
    ) -> SyscallResult<Vec<Felt>> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn call_contract(
    fn call_contract(
//        &mut self,
        &mut self,
//        _address: Felt,
        _address: Felt,
//        _entry_point_selector: Felt,
        _entry_point_selector: Felt,
//        _calldata: &[Felt],
        _calldata: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Vec<Felt>> {
    ) -> SyscallResult<Vec<Felt>> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn storage_read(
    fn storage_read(
//        &mut self,
        &mut self,
//        _address_domain: u32,
        _address_domain: u32,
//        _address: Felt,
        _address: Felt,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<Felt> {
    ) -> SyscallResult<Felt> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn storage_write(
    fn storage_write(
//        &mut self,
        &mut self,
//        _address_domain: u32,
        _address_domain: u32,
//        _address: Felt,
        _address: Felt,
//        _value: Felt,
        _value: Felt,
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn emit_event(
    fn emit_event(
//        &mut self,
        &mut self,
//        _keys: &[Felt],
        _keys: &[Felt],
//        _data: &[Felt],
        _data: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn send_message_to_l1(
    fn send_message_to_l1(
//        &mut self,
        &mut self,
//        _to_address: Felt,
        _to_address: Felt,
//        _payload: &[Felt],
        _payload: &[Felt],
//        _remaining_gas: &mut u128,
        _remaining_gas: &mut u128,
//    ) -> SyscallResult<()> {
    ) -> SyscallResult<()> {
//        unimplemented!()
        unimplemented!()
//    }
    }
//

//    fn keccak(&mut self, _input: &[u64], _remaining_gas: &mut u128) -> SyscallResult<U256> {
    fn keccak(&mut self, _input: &[u64], _remaining_gas: &mut u128) -> SyscallResult<U256> {
//        unimplemented!()
        unimplemented!()
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

//// TODO: Move to the correct place or remove if unused.
// TODO: Move to the correct place or remove if unused.
//pub(crate) mod handler {
pub(crate) mod handler {
//    use super::*;
    use super::*;
//    use std::{
    use std::{
//        alloc::Layout,
        alloc::Layout,
//        fmt::Debug,
        fmt::Debug,
//        mem::{size_of, ManuallyDrop, MaybeUninit},
        mem::{size_of, ManuallyDrop, MaybeUninit},
//        ptr::{null_mut, NonNull},
        ptr::{null_mut, NonNull},
//    };
    };
//

//    macro_rules! field_offset {
    macro_rules! field_offset {
//        ( $ident:path, $field:ident ) => {
        ( $ident:path, $field:ident ) => {
//            unsafe {
            unsafe {
//                let value_ptr = std::mem::MaybeUninit::<$ident>::uninit().as_ptr();
                let value_ptr = std::mem::MaybeUninit::<$ident>::uninit().as_ptr();
//                let field_ptr: *const u8 = std::ptr::addr_of!((*value_ptr).$field) as *const u8;
                let field_ptr: *const u8 = std::ptr::addr_of!((*value_ptr).$field) as *const u8;
//                field_ptr.offset_from(value_ptr as *const u8) as usize
                field_ptr.offset_from(value_ptr as *const u8) as usize
//            }
            }
//        };
        };
//    }
    }
//

//    #[repr(C)]
    #[repr(C)]
//    pub(crate) union SyscallResultAbi<T> {
    pub(crate) union SyscallResultAbi<T> {
//        pub ok: ManuallyDrop<SyscallResultAbiOk<T>>,
        pub ok: ManuallyDrop<SyscallResultAbiOk<T>>,
//        pub err: ManuallyDrop<SyscallResultAbiErr>,
        pub err: ManuallyDrop<SyscallResultAbiErr>,
//    }
    }
//

//    #[repr(C)]
    #[repr(C)]
//    #[derive(Debug)]
    #[derive(Debug)]
//    pub(crate) struct SyscallResultAbiOk<T> {
    pub(crate) struct SyscallResultAbiOk<T> {
//        pub tag: u8,
        pub tag: u8,
//        pub payload: ManuallyDrop<T>,
        pub payload: ManuallyDrop<T>,
//    }
    }
//

//    #[repr(C)]
    #[repr(C)]
//    #[derive(Debug)]
    #[derive(Debug)]
//    pub(crate) struct SyscallResultAbiErr {
    pub(crate) struct SyscallResultAbiErr {
//        pub tag: u8,
        pub tag: u8,
//        pub payload: ArrayAbi<Felt252Abi>,
        pub payload: ArrayAbi<Felt252Abi>,
//    }
    }
//

//    #[repr(C)]
    #[repr(C)]
//    struct ExecutionInfoAbi {
    struct ExecutionInfoAbi {
//        block_info: NonNull<BlockInfoAbi>,
        block_info: NonNull<BlockInfoAbi>,
//        tx_info: NonNull<TxInfoAbi>,
        tx_info: NonNull<TxInfoAbi>,
//        caller_address: Felt252Abi,
        caller_address: Felt252Abi,
//        contract_address: Felt252Abi,
        contract_address: Felt252Abi,
//        entry_point_selector: Felt252Abi,
        entry_point_selector: Felt252Abi,
//    }
    }
//

//    #[repr(C)]
    #[repr(C)]
//    struct ExecutionInfoV2Abi {
    struct ExecutionInfoV2Abi {
//        block_info: NonNull<BlockInfoAbi>,
        block_info: NonNull<BlockInfoAbi>,
//        tx_info: NonNull<TxInfoV2Abi>,
        tx_info: NonNull<TxInfoV2Abi>,
//        caller_address: Felt252Abi,
        caller_address: Felt252Abi,
//        contract_address: Felt252Abi,
        contract_address: Felt252Abi,
//        entry_point_selector: Felt252Abi,
        entry_point_selector: Felt252Abi,
//    }
    }
//

//    #[repr(C)]
    #[repr(C)]
//    struct TxInfoV2Abi {
    struct TxInfoV2Abi {
//        version: Felt252Abi,
        version: Felt252Abi,
//        account_contract_address: Felt252Abi,
        account_contract_address: Felt252Abi,
//        max_fee: u128,
        max_fee: u128,
//        signature: ArrayAbi<Felt252Abi>,
        signature: ArrayAbi<Felt252Abi>,
//        transaction_hash: Felt252Abi,
        transaction_hash: Felt252Abi,
//        chain_id: Felt252Abi,
        chain_id: Felt252Abi,
//        nonce: Felt252Abi,
        nonce: Felt252Abi,
//        resource_bounds: ArrayAbi<ResourceBoundsAbi>,
        resource_bounds: ArrayAbi<ResourceBoundsAbi>,
//        tip: u128,
        tip: u128,
//        paymaster_data: ArrayAbi<Felt252Abi>,
        paymaster_data: ArrayAbi<Felt252Abi>,
//        nonce_data_availability_mode: u32,
        nonce_data_availability_mode: u32,
//        fee_data_availability_mode: u32,
        fee_data_availability_mode: u32,
//        account_deployment_data: ArrayAbi<Felt252Abi>,
        account_deployment_data: ArrayAbi<Felt252Abi>,
//    }
    }
//

//    #[repr(C)]
    #[repr(C)]
//    #[derive(Debug, Clone)]
    #[derive(Debug, Clone)]
//    struct ResourceBoundsAbi {
    struct ResourceBoundsAbi {
//        resource: Felt252Abi,
        resource: Felt252Abi,
//        max_amount: u64,
        max_amount: u64,
//        max_price_per_unit: u128,
        max_price_per_unit: u128,
//    }
    }
//

//    #[repr(C)]
    #[repr(C)]
//    struct BlockInfoAbi {
    struct BlockInfoAbi {
//        block_number: u64,
        block_number: u64,
//        block_timestamp: u64,
        block_timestamp: u64,
//        sequencer_address: Felt252Abi,
        sequencer_address: Felt252Abi,
//    }
    }
//

//    #[repr(C)]
    #[repr(C)]
//    struct TxInfoAbi {
    struct TxInfoAbi {
//        version: Felt252Abi,
        version: Felt252Abi,
//        account_contract_address: Felt252Abi,
        account_contract_address: Felt252Abi,
//        max_fee: u128,
        max_fee: u128,
//        signature: ArrayAbi<Felt252Abi>,
        signature: ArrayAbi<Felt252Abi>,
//        transaction_hash: Felt252Abi,
        transaction_hash: Felt252Abi,
//        chain_id: Felt252Abi,
        chain_id: Felt252Abi,
//        nonce: Felt252Abi,
        nonce: Felt252Abi,
//    }
    }
//

//    #[repr(C)]
    #[repr(C)]
//    #[derive(Debug)]
    #[derive(Debug)]
//    pub struct StarknetSyscallHandlerCallbacks<'a, T> {
    pub struct StarknetSyscallHandlerCallbacks<'a, T> {
//        self_ptr: &'a mut T,
        self_ptr: &'a mut T,
//

//        get_block_hash: extern "C" fn(
        get_block_hash: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            block_number: u64,
            block_number: u64,
//        ),
        ),
//        get_execution_info: extern "C" fn(
        get_execution_info: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<NonNull<ExecutionInfoAbi>>,
            result_ptr: &mut SyscallResultAbi<NonNull<ExecutionInfoAbi>>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//        ),
        ),
//        get_execution_info_v2: extern "C" fn(
        get_execution_info_v2: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<NonNull<ExecutionInfoV2Abi>>,
            result_ptr: &mut SyscallResultAbi<NonNull<ExecutionInfoV2Abi>>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//        ),
        ),
//        deploy: extern "C" fn(
        deploy: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<(Felt252Abi, ArrayAbi<Felt252Abi>)>,
            result_ptr: &mut SyscallResultAbi<(Felt252Abi, ArrayAbi<Felt252Abi>)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            class_hash: &Felt252Abi,
            class_hash: &Felt252Abi,
//            contract_address_salt: &Felt252Abi,
            contract_address_salt: &Felt252Abi,
//            calldata: &ArrayAbi<Felt252Abi>,
            calldata: &ArrayAbi<Felt252Abi>,
//            deploy_from_zero: bool,
            deploy_from_zero: bool,
//        ),
        ),
//        replace_class: extern "C" fn(
        replace_class: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<()>,
            result_ptr: &mut SyscallResultAbi<()>,
//            ptr: &mut T,
            ptr: &mut T,
//            _gas: &mut u128,
            _gas: &mut u128,
//            class_hash: &Felt252Abi,
            class_hash: &Felt252Abi,
//        ),
        ),
//        library_call: extern "C" fn(
        library_call: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<ArrayAbi<Felt252Abi>>,
            result_ptr: &mut SyscallResultAbi<ArrayAbi<Felt252Abi>>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            class_hash: &Felt252Abi,
            class_hash: &Felt252Abi,
//            function_selector: &Felt252Abi,
            function_selector: &Felt252Abi,
//            calldata: &ArrayAbi<Felt252Abi>,
            calldata: &ArrayAbi<Felt252Abi>,
//        ),
        ),
//        call_contract: extern "C" fn(
        call_contract: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<ArrayAbi<Felt252Abi>>,
            result_ptr: &mut SyscallResultAbi<ArrayAbi<Felt252Abi>>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            address: &Felt252Abi,
            address: &Felt252Abi,
//            entry_point_selector: &Felt252Abi,
            entry_point_selector: &Felt252Abi,
//            calldata: &ArrayAbi<Felt252Abi>,
            calldata: &ArrayAbi<Felt252Abi>,
//        ),
        ),
//        storage_read: extern "C" fn(
        storage_read: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            address_domain: u32,
            address_domain: u32,
//            address: &Felt252Abi,
            address: &Felt252Abi,
//        ),
        ),
//        storage_write: extern "C" fn(
        storage_write: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<()>,
            result_ptr: &mut SyscallResultAbi<()>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            address_domain: u32,
            address_domain: u32,
//            address: &Felt252Abi,
            address: &Felt252Abi,
//            value: &Felt252Abi,
            value: &Felt252Abi,
//        ),
        ),
//        emit_event: extern "C" fn(
        emit_event: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<()>,
            result_ptr: &mut SyscallResultAbi<()>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            keys: &ArrayAbi<Felt252Abi>,
            keys: &ArrayAbi<Felt252Abi>,
//            data: &ArrayAbi<Felt252Abi>,
            data: &ArrayAbi<Felt252Abi>,
//        ),
        ),
//        send_message_to_l1: extern "C" fn(
        send_message_to_l1: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<()>,
            result_ptr: &mut SyscallResultAbi<()>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            to_address: &Felt252Abi,
            to_address: &Felt252Abi,
//            data: &ArrayAbi<Felt252Abi>,
            data: &ArrayAbi<Felt252Abi>,
//        ),
        ),
//        keccak: extern "C" fn(
        keccak: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<U256>,
            result_ptr: &mut SyscallResultAbi<U256>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            input: &ArrayAbi<u64>,
            input: &ArrayAbi<u64>,
//        ),
        ),
//

//        secp256k1_new: extern "C" fn(
        secp256k1_new: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256k1Point>)>,
            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256k1Point>)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            x: &U256,
            x: &U256,
//            y: &U256,
            y: &U256,
//        ),
        ),
//        secp256k1_add: extern "C" fn(
        secp256k1_add: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<Secp256k1Point>,
            result_ptr: &mut SyscallResultAbi<Secp256k1Point>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p0: &Secp256k1Point,
            p0: &Secp256k1Point,
//            p1: &Secp256k1Point,
            p1: &Secp256k1Point,
//        ),
        ),
//        secp256k1_mul: extern "C" fn(
        secp256k1_mul: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<Secp256k1Point>,
            result_ptr: &mut SyscallResultAbi<Secp256k1Point>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p: &Secp256k1Point,
            p: &Secp256k1Point,
//            scalar: &U256,
            scalar: &U256,
//        ),
        ),
//        secp256k1_get_point_from_x: extern "C" fn(
        secp256k1_get_point_from_x: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256k1Point>)>,
            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256k1Point>)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            x: &U256,
            x: &U256,
//            y_parity: &bool,
            y_parity: &bool,
//        ),
        ),
//        secp256k1_get_xy: extern "C" fn(
        secp256k1_get_xy: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<(U256, U256)>,
            result_ptr: &mut SyscallResultAbi<(U256, U256)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p: &Secp256k1Point,
            p: &Secp256k1Point,
//        ),
        ),
//

//        secp256r1_new: extern "C" fn(
        secp256r1_new: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256r1Point>)>,
            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256r1Point>)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            x: &U256,
            x: &U256,
//            y: &U256,
            y: &U256,
//        ),
        ),
//        secp256r1_add: extern "C" fn(
        secp256r1_add: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<Secp256r1Point>,
            result_ptr: &mut SyscallResultAbi<Secp256r1Point>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p0: &Secp256r1Point,
            p0: &Secp256r1Point,
//            p1: &Secp256r1Point,
            p1: &Secp256r1Point,
//        ),
        ),
//        secp256r1_mul: extern "C" fn(
        secp256r1_mul: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<Secp256r1Point>,
            result_ptr: &mut SyscallResultAbi<Secp256r1Point>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p: &Secp256r1Point,
            p: &Secp256r1Point,
//            scalar: &U256,
            scalar: &U256,
//        ),
        ),
//        secp256r1_get_point_from_x: extern "C" fn(
        secp256r1_get_point_from_x: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256r1Point>)>,
            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256r1Point>)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            x: &U256,
            x: &U256,
//            y_parity: &bool,
            y_parity: &bool,
//        ),
        ),
//        secp256r1_get_xy: extern "C" fn(
        secp256r1_get_xy: extern "C" fn(
//            result_ptr: &mut SyscallResultAbi<(U256, U256)>,
            result_ptr: &mut SyscallResultAbi<(U256, U256)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p: &Secp256r1Point,
            p: &Secp256r1Point,
//        ),
        ),
//    }
    }
//

//    impl<'a, T> StarknetSyscallHandlerCallbacks<'a, T>
    impl<'a, T> StarknetSyscallHandlerCallbacks<'a, T>
//    where
    where
//        T: 'a,
        T: 'a,
//    {
    {
//        // Callback field indices.
        // Callback field indices.
//        pub const CALL_CONTRACT: usize = field_offset!(Self, call_contract) >> 3;
        pub const CALL_CONTRACT: usize = field_offset!(Self, call_contract) >> 3;
//        pub const DEPLOY: usize = field_offset!(Self, deploy) >> 3;
        pub const DEPLOY: usize = field_offset!(Self, deploy) >> 3;
//        pub const EMIT_EVENT: usize = field_offset!(Self, emit_event) >> 3;
        pub const EMIT_EVENT: usize = field_offset!(Self, emit_event) >> 3;
//        pub const GET_BLOCK_HASH: usize = field_offset!(Self, get_block_hash) >> 3;
        pub const GET_BLOCK_HASH: usize = field_offset!(Self, get_block_hash) >> 3;
//        pub const GET_EXECUTION_INFO: usize = field_offset!(Self, get_execution_info) >> 3;
        pub const GET_EXECUTION_INFO: usize = field_offset!(Self, get_execution_info) >> 3;
//        pub const GET_EXECUTION_INFOV2: usize = field_offset!(Self, get_execution_info_v2) >> 3;
        pub const GET_EXECUTION_INFOV2: usize = field_offset!(Self, get_execution_info_v2) >> 3;
//        pub const KECCAK: usize = field_offset!(Self, keccak) >> 3;
        pub const KECCAK: usize = field_offset!(Self, keccak) >> 3;
//        pub const LIBRARY_CALL: usize = field_offset!(Self, library_call) >> 3;
        pub const LIBRARY_CALL: usize = field_offset!(Self, library_call) >> 3;
//        pub const REPLACE_CLASS: usize = field_offset!(Self, replace_class) >> 3;
        pub const REPLACE_CLASS: usize = field_offset!(Self, replace_class) >> 3;
//        pub const SEND_MESSAGE_TO_L1: usize = field_offset!(Self, send_message_to_l1) >> 3;
        pub const SEND_MESSAGE_TO_L1: usize = field_offset!(Self, send_message_to_l1) >> 3;
//        pub const STORAGE_READ: usize = field_offset!(Self, storage_read) >> 3;
        pub const STORAGE_READ: usize = field_offset!(Self, storage_read) >> 3;
//        pub const STORAGE_WRITE: usize = field_offset!(Self, storage_write) >> 3;
        pub const STORAGE_WRITE: usize = field_offset!(Self, storage_write) >> 3;
//

//        pub const SECP256K1_NEW: usize = field_offset!(Self, secp256k1_new) >> 3;
        pub const SECP256K1_NEW: usize = field_offset!(Self, secp256k1_new) >> 3;
//        pub const SECP256K1_ADD: usize = field_offset!(Self, secp256k1_add) >> 3;
        pub const SECP256K1_ADD: usize = field_offset!(Self, secp256k1_add) >> 3;
//        pub const SECP256K1_MUL: usize = field_offset!(Self, secp256k1_mul) >> 3;
        pub const SECP256K1_MUL: usize = field_offset!(Self, secp256k1_mul) >> 3;
//        pub const SECP256K1_GET_POINT_FROM_X: usize =
        pub const SECP256K1_GET_POINT_FROM_X: usize =
//            field_offset!(Self, secp256k1_get_point_from_x) >> 3;
            field_offset!(Self, secp256k1_get_point_from_x) >> 3;
//        pub const SECP256K1_GET_XY: usize = field_offset!(Self, secp256k1_get_xy) >> 3;
        pub const SECP256K1_GET_XY: usize = field_offset!(Self, secp256k1_get_xy) >> 3;
//        pub const SECP256R1_NEW: usize = field_offset!(Self, secp256r1_new) >> 3;
        pub const SECP256R1_NEW: usize = field_offset!(Self, secp256r1_new) >> 3;
//        pub const SECP256R1_ADD: usize = field_offset!(Self, secp256r1_add) >> 3;
        pub const SECP256R1_ADD: usize = field_offset!(Self, secp256r1_add) >> 3;
//        pub const SECP256R1_MUL: usize = field_offset!(Self, secp256r1_mul) >> 3;
        pub const SECP256R1_MUL: usize = field_offset!(Self, secp256r1_mul) >> 3;
//        pub const SECP256R1_GET_POINT_FROM_X: usize =
        pub const SECP256R1_GET_POINT_FROM_X: usize =
//            field_offset!(Self, secp256r1_get_point_from_x) >> 3;
            field_offset!(Self, secp256r1_get_point_from_x) >> 3;
//        pub const SECP256R1_GET_XY: usize = field_offset!(Self, secp256r1_get_xy) >> 3;
        pub const SECP256R1_GET_XY: usize = field_offset!(Self, secp256r1_get_xy) >> 3;
//    }
    }
//

//    #[allow(unused_variables)]
    #[allow(unused_variables)]
//    impl<'a, T> StarknetSyscallHandlerCallbacks<'a, T>
    impl<'a, T> StarknetSyscallHandlerCallbacks<'a, T>
//    where
    where
//        T: StarknetSyscallHandler + 'a,
        T: StarknetSyscallHandler + 'a,
//    {
    {
//        pub fn new(handler: &'a mut T) -> Self {
        pub fn new(handler: &'a mut T) -> Self {
//            Self {
            Self {
//                self_ptr: handler,
                self_ptr: handler,
//                get_block_hash: Self::wrap_get_block_hash,
                get_block_hash: Self::wrap_get_block_hash,
//                get_execution_info: Self::wrap_get_execution_info,
                get_execution_info: Self::wrap_get_execution_info,
//                get_execution_info_v2: Self::wrap_get_execution_info_v2,
                get_execution_info_v2: Self::wrap_get_execution_info_v2,
//                deploy: Self::wrap_deploy,
                deploy: Self::wrap_deploy,
//                replace_class: Self::wrap_replace_class,
                replace_class: Self::wrap_replace_class,
//                library_call: Self::wrap_library_call,
                library_call: Self::wrap_library_call,
//                call_contract: Self::wrap_call_contract,
                call_contract: Self::wrap_call_contract,
//                storage_read: Self::wrap_storage_read,
                storage_read: Self::wrap_storage_read,
//                storage_write: Self::wrap_storage_write,
                storage_write: Self::wrap_storage_write,
//                emit_event: Self::wrap_emit_event,
                emit_event: Self::wrap_emit_event,
//                send_message_to_l1: Self::wrap_send_message_to_l1,
                send_message_to_l1: Self::wrap_send_message_to_l1,
//                keccak: Self::wrap_keccak,
                keccak: Self::wrap_keccak,
//                secp256k1_new: Self::wrap_secp256k1_new,
                secp256k1_new: Self::wrap_secp256k1_new,
//                secp256k1_add: Self::wrap_secp256k1_add,
                secp256k1_add: Self::wrap_secp256k1_add,
//                secp256k1_mul: Self::wrap_secp256k1_mul,
                secp256k1_mul: Self::wrap_secp256k1_mul,
//                secp256k1_get_point_from_x: Self::wrap_secp256k1_get_point_from_x,
                secp256k1_get_point_from_x: Self::wrap_secp256k1_get_point_from_x,
//                secp256k1_get_xy: Self::wrap_secp256k1_get_xy,
                secp256k1_get_xy: Self::wrap_secp256k1_get_xy,
//                secp256r1_new: Self::wrap_secp256r1_new,
                secp256r1_new: Self::wrap_secp256r1_new,
//                secp256r1_add: Self::wrap_secp256r1_add,
                secp256r1_add: Self::wrap_secp256r1_add,
//                secp256r1_mul: Self::wrap_secp256r1_mul,
                secp256r1_mul: Self::wrap_secp256r1_mul,
//                secp256r1_get_point_from_x: Self::wrap_secp256r1_get_point_from_x,
                secp256r1_get_point_from_x: Self::wrap_secp256r1_get_point_from_x,
//                secp256r1_get_xy: Self::wrap_secp256r1_get_xy,
                secp256r1_get_xy: Self::wrap_secp256r1_get_xy,
//            }
            }
//        }
        }
//

//        unsafe fn alloc_mlir_array<E: Clone>(data: &[E]) -> ArrayAbi<E> {
        unsafe fn alloc_mlir_array<E: Clone>(data: &[E]) -> ArrayAbi<E> {
//            match data.len() {
            match data.len() {
//                0 => ArrayAbi {
                0 => ArrayAbi {
//                    ptr: null_mut(),
                    ptr: null_mut(),
//                    since: 0,
                    since: 0,
//                    until: 0,
                    until: 0,
//                    capacity: 0,
                    capacity: 0,
//                },
                },
//                _ => {
                _ => {
//                    let ptr =
                    let ptr =
//                        libc::malloc(Layout::array::<E>(data.len()).unwrap().size()) as *mut E;
                        libc::malloc(Layout::array::<E>(data.len()).unwrap().size()) as *mut E;
//

//                    let len: u32 = data.len().try_into().unwrap();
                    let len: u32 = data.len().try_into().unwrap();
//                    for (i, val) in data.iter().enumerate() {
                    for (i, val) in data.iter().enumerate() {
//                        ptr.add(i).write(val.clone());
                        ptr.add(i).write(val.clone());
//                    }
                    }
//

//                    ArrayAbi {
                    ArrayAbi {
//                        ptr,
                        ptr,
//                        since: 0,
                        since: 0,
//                        until: len,
                        until: len,
//                        capacity: len,
                        capacity: len,
//                    }
                    }
//                }
                }
//            }
            }
//        }
        }
//

//        fn wrap_error<E>(e: &[Felt]) -> SyscallResultAbi<E> {
        fn wrap_error<E>(e: &[Felt]) -> SyscallResultAbi<E> {
//            SyscallResultAbi {
            SyscallResultAbi {
//                err: ManuallyDrop::new(SyscallResultAbiErr {
                err: ManuallyDrop::new(SyscallResultAbiErr {
//                    tag: 1u8,
                    tag: 1u8,
//                    payload: unsafe {
                    payload: unsafe {
//                        let data: Vec<_> = e.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
                        let data: Vec<_> = e.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
//                        Self::alloc_mlir_array(&data)
                        Self::alloc_mlir_array(&data)
//                    },
                    },
//                }),
                }),
//            }
            }
//        }
        }
//

//        extern "C" fn wrap_get_block_hash(
        extern "C" fn wrap_get_block_hash(
//            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            block_number: u64,
            block_number: u64,
//        ) {
        ) {
//            let result = ptr.get_block_hash(block_number, gas);
            let result = ptr.get_block_hash(block_number, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(Felt252Abi(x.to_bytes_le())),
                        payload: ManuallyDrop::new(Felt252Abi(x.to_bytes_le())),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_get_execution_info(
        extern "C" fn wrap_get_execution_info(
//            result_ptr: &mut SyscallResultAbi<NonNull<ExecutionInfoAbi>>,
            result_ptr: &mut SyscallResultAbi<NonNull<ExecutionInfoAbi>>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//        ) {
        ) {
//            let result = ptr.get_execution_info(gas);
            let result = ptr.get_execution_info(gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: unsafe {
                        payload: unsafe {
//                            let mut block_info_ptr =
                            let mut block_info_ptr =
//                                NonNull::new(
                                NonNull::new(
//                                    libc::malloc(size_of::<BlockInfoAbi>()) as *mut BlockInfoAbi
                                    libc::malloc(size_of::<BlockInfoAbi>()) as *mut BlockInfoAbi
//                                )
                                )
//                                .unwrap();
                                .unwrap();
//                            block_info_ptr.as_mut().block_number = x.block_info.block_number;
                            block_info_ptr.as_mut().block_number = x.block_info.block_number;
//                            block_info_ptr.as_mut().block_timestamp = x.block_info.block_timestamp;
                            block_info_ptr.as_mut().block_timestamp = x.block_info.block_timestamp;
//                            block_info_ptr.as_mut().sequencer_address =
                            block_info_ptr.as_mut().sequencer_address =
//                                Felt252Abi(x.block_info.sequencer_address.to_bytes_le());
                                Felt252Abi(x.block_info.sequencer_address.to_bytes_le());
//

//                            let mut tx_info_ptr = NonNull::new(
                            let mut tx_info_ptr = NonNull::new(
//                                libc::malloc(size_of::<TxInfoAbi>()) as *mut TxInfoAbi,
                                libc::malloc(size_of::<TxInfoAbi>()) as *mut TxInfoAbi,
//                            )
                            )
//                            .unwrap();
                            .unwrap();
//                            tx_info_ptr.as_mut().version =
                            tx_info_ptr.as_mut().version =
//                                Felt252Abi(x.tx_info.version.to_bytes_le());
                                Felt252Abi(x.tx_info.version.to_bytes_le());
//                            tx_info_ptr.as_mut().account_contract_address =
                            tx_info_ptr.as_mut().account_contract_address =
//                                Felt252Abi(x.tx_info.account_contract_address.to_bytes_le());
                                Felt252Abi(x.tx_info.account_contract_address.to_bytes_le());
//                            tx_info_ptr.as_mut().max_fee = x.tx_info.max_fee;
                            tx_info_ptr.as_mut().max_fee = x.tx_info.max_fee;
//                            tx_info_ptr.as_mut().signature = Self::alloc_mlir_array(
                            tx_info_ptr.as_mut().signature = Self::alloc_mlir_array(
//                                &x.tx_info
                                &x.tx_info
//                                    .signature
                                    .signature
//                                    .into_iter()
                                    .into_iter()
//                                    .map(|x| Felt252Abi(x.to_bytes_le()))
                                    .map(|x| Felt252Abi(x.to_bytes_le()))
//                                    .collect::<Vec<_>>(),
                                    .collect::<Vec<_>>(),
//                            );
                            );
//                            tx_info_ptr.as_mut().transaction_hash =
                            tx_info_ptr.as_mut().transaction_hash =
//                                Felt252Abi(x.tx_info.transaction_hash.to_bytes_le());
                                Felt252Abi(x.tx_info.transaction_hash.to_bytes_le());
//                            tx_info_ptr.as_mut().chain_id =
                            tx_info_ptr.as_mut().chain_id =
//                                Felt252Abi(x.tx_info.chain_id.to_bytes_le());
                                Felt252Abi(x.tx_info.chain_id.to_bytes_le());
//                            tx_info_ptr.as_mut().nonce = Felt252Abi(x.tx_info.nonce.to_bytes_le());
                            tx_info_ptr.as_mut().nonce = Felt252Abi(x.tx_info.nonce.to_bytes_le());
//

//                            let mut execution_info_ptr =
                            let mut execution_info_ptr =
//                                NonNull::new(libc::malloc(size_of::<ExecutionInfoAbi>())
                                NonNull::new(libc::malloc(size_of::<ExecutionInfoAbi>())
//                                    as *mut ExecutionInfoAbi)
                                    as *mut ExecutionInfoAbi)
//                                .unwrap();
                                .unwrap();
//                            execution_info_ptr.as_mut().block_info = block_info_ptr;
                            execution_info_ptr.as_mut().block_info = block_info_ptr;
//                            execution_info_ptr.as_mut().tx_info = tx_info_ptr;
                            execution_info_ptr.as_mut().tx_info = tx_info_ptr;
//                            execution_info_ptr.as_mut().caller_address =
                            execution_info_ptr.as_mut().caller_address =
//                                Felt252Abi(x.caller_address.to_bytes_le());
                                Felt252Abi(x.caller_address.to_bytes_le());
//                            execution_info_ptr.as_mut().contract_address =
                            execution_info_ptr.as_mut().contract_address =
//                                Felt252Abi(x.contract_address.to_bytes_le());
                                Felt252Abi(x.contract_address.to_bytes_le());
//                            execution_info_ptr.as_mut().entry_point_selector =
                            execution_info_ptr.as_mut().entry_point_selector =
//                                Felt252Abi(x.entry_point_selector.to_bytes_le());
                                Felt252Abi(x.entry_point_selector.to_bytes_le());
//

//                            ManuallyDrop::new(execution_info_ptr)
                            ManuallyDrop::new(execution_info_ptr)
//                        },
                        },
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_get_execution_info_v2(
        extern "C" fn wrap_get_execution_info_v2(
//            result_ptr: &mut SyscallResultAbi<NonNull<ExecutionInfoV2Abi>>,
            result_ptr: &mut SyscallResultAbi<NonNull<ExecutionInfoV2Abi>>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//        ) {
        ) {
//            let result = ptr.get_execution_info_v2(gas);
            let result = ptr.get_execution_info_v2(gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: unsafe {
                        payload: unsafe {
//                            let mut execution_info_ptr =
                            let mut execution_info_ptr =
//                                NonNull::new(libc::malloc(size_of::<ExecutionInfoV2Abi>())
                                NonNull::new(libc::malloc(size_of::<ExecutionInfoV2Abi>())
//                                    as *mut ExecutionInfoV2Abi)
                                    as *mut ExecutionInfoV2Abi)
//                                .unwrap();
                                .unwrap();
//

//                            let mut block_info_ptr =
                            let mut block_info_ptr =
//                                NonNull::new(
                                NonNull::new(
//                                    libc::malloc(size_of::<BlockInfoAbi>()) as *mut BlockInfoAbi
                                    libc::malloc(size_of::<BlockInfoAbi>()) as *mut BlockInfoAbi
//                                )
                                )
//                                .unwrap();
                                .unwrap();
//                            block_info_ptr.as_mut().block_number = x.block_info.block_number;
                            block_info_ptr.as_mut().block_number = x.block_info.block_number;
//                            block_info_ptr.as_mut().block_timestamp = x.block_info.block_timestamp;
                            block_info_ptr.as_mut().block_timestamp = x.block_info.block_timestamp;
//                            block_info_ptr.as_mut().sequencer_address =
                            block_info_ptr.as_mut().sequencer_address =
//                                Felt252Abi(x.block_info.sequencer_address.to_bytes_le());
                                Felt252Abi(x.block_info.sequencer_address.to_bytes_le());
//

//                            let mut tx_info_ptr = NonNull::new(
                            let mut tx_info_ptr = NonNull::new(
//                                libc::malloc(size_of::<TxInfoV2Abi>()) as *mut TxInfoV2Abi,
                                libc::malloc(size_of::<TxInfoV2Abi>()) as *mut TxInfoV2Abi,
//                            )
                            )
//                            .unwrap();
                            .unwrap();
//                            tx_info_ptr.as_mut().version =
                            tx_info_ptr.as_mut().version =
//                                Felt252Abi(x.tx_info.version.to_bytes_le());
                                Felt252Abi(x.tx_info.version.to_bytes_le());
//                            tx_info_ptr.as_mut().signature = Self::alloc_mlir_array(
                            tx_info_ptr.as_mut().signature = Self::alloc_mlir_array(
//                                &x.tx_info
                                &x.tx_info
//                                    .signature
                                    .signature
//                                    .into_iter()
                                    .into_iter()
//                                    .map(|x| Felt252Abi(x.to_bytes_le()))
                                    .map(|x| Felt252Abi(x.to_bytes_le()))
//                                    .collect::<Vec<_>>(),
                                    .collect::<Vec<_>>(),
//                            );
                            );
//                            tx_info_ptr.as_mut().max_fee = x.tx_info.max_fee;
                            tx_info_ptr.as_mut().max_fee = x.tx_info.max_fee;
//                            tx_info_ptr.as_mut().transaction_hash =
                            tx_info_ptr.as_mut().transaction_hash =
//                                Felt252Abi(x.tx_info.transaction_hash.to_bytes_le());
                                Felt252Abi(x.tx_info.transaction_hash.to_bytes_le());
//                            tx_info_ptr.as_mut().chain_id =
                            tx_info_ptr.as_mut().chain_id =
//                                Felt252Abi(x.tx_info.chain_id.to_bytes_le());
                                Felt252Abi(x.tx_info.chain_id.to_bytes_le());
//                            tx_info_ptr.as_mut().nonce = Felt252Abi(x.tx_info.nonce.to_bytes_le());
                            tx_info_ptr.as_mut().nonce = Felt252Abi(x.tx_info.nonce.to_bytes_le());
//                            tx_info_ptr.as_mut().resource_bounds = Self::alloc_mlir_array(
                            tx_info_ptr.as_mut().resource_bounds = Self::alloc_mlir_array(
//                                &x.tx_info
                                &x.tx_info
//                                    .resource_bounds
                                    .resource_bounds
//                                    .into_iter()
                                    .into_iter()
//                                    .map(|x| ResourceBoundsAbi {
                                    .map(|x| ResourceBoundsAbi {
//                                        resource: Felt252Abi(x.resource.to_bytes_le()),
                                        resource: Felt252Abi(x.resource.to_bytes_le()),
//                                        max_amount: x.max_amount,
                                        max_amount: x.max_amount,
//                                        max_price_per_unit: x.max_price_per_unit,
                                        max_price_per_unit: x.max_price_per_unit,
//                                    })
                                    })
//                                    .collect::<Vec<_>>(),
                                    .collect::<Vec<_>>(),
//                            );
                            );
//                            tx_info_ptr.as_mut().tip = x.tx_info.tip;
                            tx_info_ptr.as_mut().tip = x.tx_info.tip;
//                            tx_info_ptr.as_mut().paymaster_data = Self::alloc_mlir_array(
                            tx_info_ptr.as_mut().paymaster_data = Self::alloc_mlir_array(
//                                &x.tx_info
                                &x.tx_info
//                                    .paymaster_data
                                    .paymaster_data
//                                    .into_iter()
                                    .into_iter()
//                                    .map(|x| Felt252Abi(x.to_bytes_le()))
                                    .map(|x| Felt252Abi(x.to_bytes_le()))
//                                    .collect::<Vec<_>>(),
                                    .collect::<Vec<_>>(),
//                            );
                            );
//                            tx_info_ptr.as_mut().nonce_data_availability_mode =
                            tx_info_ptr.as_mut().nonce_data_availability_mode =
//                                x.tx_info.nonce_data_availability_mode;
                                x.tx_info.nonce_data_availability_mode;
//                            tx_info_ptr.as_mut().fee_data_availability_mode =
                            tx_info_ptr.as_mut().fee_data_availability_mode =
//                                x.tx_info.fee_data_availability_mode;
                                x.tx_info.fee_data_availability_mode;
//                            tx_info_ptr.as_mut().account_deployment_data = Self::alloc_mlir_array(
                            tx_info_ptr.as_mut().account_deployment_data = Self::alloc_mlir_array(
//                                &x.tx_info
                                &x.tx_info
//                                    .account_deployment_data
                                    .account_deployment_data
//                                    .into_iter()
                                    .into_iter()
//                                    .map(|x| Felt252Abi(x.to_bytes_le()))
                                    .map(|x| Felt252Abi(x.to_bytes_le()))
//                                    .collect::<Vec<_>>(),
                                    .collect::<Vec<_>>(),
//                            );
                            );
//                            tx_info_ptr.as_mut().account_contract_address =
                            tx_info_ptr.as_mut().account_contract_address =
//                                Felt252Abi(x.tx_info.account_contract_address.to_bytes_le());
                                Felt252Abi(x.tx_info.account_contract_address.to_bytes_le());
//

//                            execution_info_ptr.as_mut().block_info = block_info_ptr;
                            execution_info_ptr.as_mut().block_info = block_info_ptr;
//                            execution_info_ptr.as_mut().tx_info = tx_info_ptr;
                            execution_info_ptr.as_mut().tx_info = tx_info_ptr;
//                            execution_info_ptr.as_mut().caller_address =
                            execution_info_ptr.as_mut().caller_address =
//                                Felt252Abi(x.caller_address.to_bytes_le());
                                Felt252Abi(x.caller_address.to_bytes_le());
//                            execution_info_ptr.as_mut().contract_address =
                            execution_info_ptr.as_mut().contract_address =
//                                Felt252Abi(x.contract_address.to_bytes_le());
                                Felt252Abi(x.contract_address.to_bytes_le());
//                            execution_info_ptr.as_mut().entry_point_selector =
                            execution_info_ptr.as_mut().entry_point_selector =
//                                Felt252Abi(x.entry_point_selector.to_bytes_le());
                                Felt252Abi(x.entry_point_selector.to_bytes_le());
//

//                            ManuallyDrop::new(execution_info_ptr)
                            ManuallyDrop::new(execution_info_ptr)
//                        },
                        },
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        // TODO: change all from_bytes_be to from_bytes_ne when added and undo byte swapping.
        // TODO: change all from_bytes_be to from_bytes_ne when added and undo byte swapping.
//

//        extern "C" fn wrap_deploy(
        extern "C" fn wrap_deploy(
//            result_ptr: &mut SyscallResultAbi<(Felt252Abi, ArrayAbi<Felt252Abi>)>,
            result_ptr: &mut SyscallResultAbi<(Felt252Abi, ArrayAbi<Felt252Abi>)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            class_hash: &Felt252Abi,
            class_hash: &Felt252Abi,
//            contract_address_salt: &Felt252Abi,
            contract_address_salt: &Felt252Abi,
//            calldata: &ArrayAbi<Felt252Abi>,
            calldata: &ArrayAbi<Felt252Abi>,
//            deploy_from_zero: bool,
            deploy_from_zero: bool,
//        ) {
        ) {
//            let class_hash = Felt::from_bytes_be(&{
            let class_hash = Felt::from_bytes_be(&{
//                let mut data = class_hash.0;
                let mut data = class_hash.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//            let contract_address_salt = Felt::from_bytes_be(&{
            let contract_address_salt = Felt::from_bytes_be(&{
//                let mut data = contract_address_salt.0;
                let mut data = contract_address_salt.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//

//            let calldata: Vec<_> = unsafe {
            let calldata: Vec<_> = unsafe {
//                let since_offset = calldata.since as usize;
                let since_offset = calldata.since as usize;
//                let until_offset = calldata.until as usize;
                let until_offset = calldata.until as usize;
//                debug_assert!(since_offset <= until_offset);
                debug_assert!(since_offset <= until_offset);
//                let len = until_offset - since_offset;
                let len = until_offset - since_offset;
//                match len {
                match len {
//                    0 => &[],
                    0 => &[],
//                    _ => std::slice::from_raw_parts(calldata.ptr.add(since_offset), len),
                    _ => std::slice::from_raw_parts(calldata.ptr.add(since_offset), len),
//                }
                }
//            }
            }
//            .iter()
            .iter()
//            .map(|x| {
            .map(|x| {
//                Felt::from_bytes_be(&{
                Felt::from_bytes_be(&{
//                    let mut data = x.0;
                    let mut data = x.0;
//                    data.reverse();
                    data.reverse();
//                    data
                    data
//                })
                })
//            })
            })
//            .collect();
            .collect();
//

//            let result = ptr.deploy(
            let result = ptr.deploy(
//                class_hash,
                class_hash,
//                contract_address_salt,
                contract_address_salt,
//                &calldata,
                &calldata,
//                deploy_from_zero,
                deploy_from_zero,
//                gas,
                gas,
//            );
            );
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => {
                Ok(x) => {
//                    let felts: Vec<_> = x.1.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
                    let felts: Vec<_> = x.1.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
//                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
//                    SyscallResultAbi {
                    SyscallResultAbi {
//                        ok: ManuallyDrop::new(SyscallResultAbiOk {
                        ok: ManuallyDrop::new(SyscallResultAbiOk {
//                            tag: 0u8,
                            tag: 0u8,
//                            payload: ManuallyDrop::new((Felt252Abi(x.0.to_bytes_le()), felts_ptr)),
                            payload: ManuallyDrop::new((Felt252Abi(x.0.to_bytes_le()), felts_ptr)),
//                        }),
                        }),
//                    }
                    }
//                }
                }
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_replace_class(
        extern "C" fn wrap_replace_class(
//            result_ptr: &mut SyscallResultAbi<()>,
            result_ptr: &mut SyscallResultAbi<()>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            class_hash: &Felt252Abi,
            class_hash: &Felt252Abi,
//        ) {
        ) {
//            let class_hash = Felt::from_bytes_be(&{
            let class_hash = Felt::from_bytes_be(&{
//                let mut data = class_hash.0;
                let mut data = class_hash.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//            let result = ptr.replace_class(class_hash, gas);
            let result = ptr.replace_class(class_hash, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(_) => SyscallResultAbi {
                Ok(_) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(()),
                        payload: ManuallyDrop::new(()),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_library_call(
        extern "C" fn wrap_library_call(
//            result_ptr: &mut SyscallResultAbi<ArrayAbi<Felt252Abi>>,
            result_ptr: &mut SyscallResultAbi<ArrayAbi<Felt252Abi>>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            class_hash: &Felt252Abi,
            class_hash: &Felt252Abi,
//            function_selector: &Felt252Abi,
            function_selector: &Felt252Abi,
//            calldata: &ArrayAbi<Felt252Abi>,
            calldata: &ArrayAbi<Felt252Abi>,
//        ) {
        ) {
//            let class_hash = Felt::from_bytes_be(&{
            let class_hash = Felt::from_bytes_be(&{
//                let mut data = class_hash.0;
                let mut data = class_hash.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//            let function_selector = Felt::from_bytes_be(&{
            let function_selector = Felt::from_bytes_be(&{
//                let mut data = function_selector.0;
                let mut data = function_selector.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//

//            let calldata: Vec<_> = unsafe {
            let calldata: Vec<_> = unsafe {
//                let since_offset = calldata.since as usize;
                let since_offset = calldata.since as usize;
//                let until_offset = calldata.until as usize;
                let until_offset = calldata.until as usize;
//                debug_assert!(since_offset <= until_offset);
                debug_assert!(since_offset <= until_offset);
//                let len = until_offset - since_offset;
                let len = until_offset - since_offset;
//                match len {
                match len {
//                    0 => &[],
                    0 => &[],
//                    _ => std::slice::from_raw_parts(calldata.ptr.add(since_offset), len),
                    _ => std::slice::from_raw_parts(calldata.ptr.add(since_offset), len),
//                }
                }
//            }
            }
//            .iter()
            .iter()
//            .map(|x| {
            .map(|x| {
//                Felt::from_bytes_be(&{
                Felt::from_bytes_be(&{
//                    let mut data = x.0;
                    let mut data = x.0;
//                    data.reverse();
                    data.reverse();
//                    data
                    data
//                })
                })
//            })
            })
//            .collect();
            .collect();
//

//            let result = ptr.library_call(class_hash, function_selector, &calldata, gas);
            let result = ptr.library_call(class_hash, function_selector, &calldata, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => {
                Ok(x) => {
//                    let felts: Vec<_> = x.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
                    let felts: Vec<_> = x.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
//                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
//                    SyscallResultAbi {
                    SyscallResultAbi {
//                        ok: ManuallyDrop::new(SyscallResultAbiOk {
                        ok: ManuallyDrop::new(SyscallResultAbiOk {
//                            tag: 0u8,
                            tag: 0u8,
//                            payload: ManuallyDrop::new(felts_ptr),
                            payload: ManuallyDrop::new(felts_ptr),
//                        }),
                        }),
//                    }
                    }
//                }
                }
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_call_contract(
        extern "C" fn wrap_call_contract(
//            result_ptr: &mut SyscallResultAbi<ArrayAbi<Felt252Abi>>,
            result_ptr: &mut SyscallResultAbi<ArrayAbi<Felt252Abi>>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            address: &Felt252Abi,
            address: &Felt252Abi,
//            entry_point_selector: &Felt252Abi,
            entry_point_selector: &Felt252Abi,
//            calldata: &ArrayAbi<Felt252Abi>,
            calldata: &ArrayAbi<Felt252Abi>,
//        ) {
        ) {
//            let address = Felt::from_bytes_be(&{
            let address = Felt::from_bytes_be(&{
//                let mut data = address.0;
                let mut data = address.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//            let entry_point_selector = Felt::from_bytes_be(&{
            let entry_point_selector = Felt::from_bytes_be(&{
//                let mut data = entry_point_selector.0;
                let mut data = entry_point_selector.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//

//            let calldata: Vec<_> = unsafe {
            let calldata: Vec<_> = unsafe {
//                let since_offset = calldata.since as usize;
                let since_offset = calldata.since as usize;
//                let until_offset = calldata.until as usize;
                let until_offset = calldata.until as usize;
//                debug_assert!(since_offset <= until_offset);
                debug_assert!(since_offset <= until_offset);
//                let len = until_offset - since_offset;
                let len = until_offset - since_offset;
//                match len {
                match len {
//                    0 => &[],
                    0 => &[],
//                    _ => std::slice::from_raw_parts(calldata.ptr.add(since_offset), len),
                    _ => std::slice::from_raw_parts(calldata.ptr.add(since_offset), len),
//                }
                }
//            }
            }
//            .iter()
            .iter()
//            .map(|x| {
            .map(|x| {
//                Felt::from_bytes_be(&{
                Felt::from_bytes_be(&{
//                    let mut data = x.0;
                    let mut data = x.0;
//                    data.reverse();
                    data.reverse();
//                    data
                    data
//                })
                })
//            })
            })
//            .collect();
            .collect();
//

//            let result = ptr.call_contract(address, entry_point_selector, &calldata, gas);
            let result = ptr.call_contract(address, entry_point_selector, &calldata, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => {
                Ok(x) => {
//                    let felts: Vec<_> = x.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
                    let felts: Vec<_> = x.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
//                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
//                    SyscallResultAbi {
                    SyscallResultAbi {
//                        ok: ManuallyDrop::new(SyscallResultAbiOk {
                        ok: ManuallyDrop::new(SyscallResultAbiOk {
//                            tag: 0u8,
                            tag: 0u8,
//                            payload: ManuallyDrop::new(felts_ptr),
                            payload: ManuallyDrop::new(felts_ptr),
//                        }),
                        }),
//                    }
                    }
//                }
                }
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_storage_read(
        extern "C" fn wrap_storage_read(
//            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            address_domain: u32,
            address_domain: u32,
//            address: &Felt252Abi,
            address: &Felt252Abi,
//        ) {
        ) {
//            let address = Felt::from_bytes_be(&{
            let address = Felt::from_bytes_be(&{
//                let mut data = address.0;
                let mut data = address.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//            let result = ptr.storage_read(address_domain, address, gas);
            let result = ptr.storage_read(address_domain, address, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(res) => SyscallResultAbi {
                Ok(res) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(Felt252Abi(res.to_bytes_le())),
                        payload: ManuallyDrop::new(Felt252Abi(res.to_bytes_le())),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_storage_write(
        extern "C" fn wrap_storage_write(
//            result_ptr: &mut SyscallResultAbi<()>,
            result_ptr: &mut SyscallResultAbi<()>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            address_domain: u32,
            address_domain: u32,
//            address: &Felt252Abi,
            address: &Felt252Abi,
//            value: &Felt252Abi,
            value: &Felt252Abi,
//        ) {
        ) {
//            let address = Felt::from_bytes_be(&{
            let address = Felt::from_bytes_be(&{
//                let mut data = address.0;
                let mut data = address.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//            let value = Felt::from_bytes_be(&{
            let value = Felt::from_bytes_be(&{
//                let mut data = value.0;
                let mut data = value.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//            let result = ptr.storage_write(address_domain, address, value, gas);
            let result = ptr.storage_write(address_domain, address, value, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(_) => SyscallResultAbi {
                Ok(_) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(()),
                        payload: ManuallyDrop::new(()),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_emit_event(
        extern "C" fn wrap_emit_event(
//            result_ptr: &mut SyscallResultAbi<()>,
            result_ptr: &mut SyscallResultAbi<()>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            keys: &ArrayAbi<Felt252Abi>,
            keys: &ArrayAbi<Felt252Abi>,
//            data: &ArrayAbi<Felt252Abi>,
            data: &ArrayAbi<Felt252Abi>,
//        ) {
        ) {
//            let keys: Vec<_> = unsafe {
            let keys: Vec<_> = unsafe {
//                let since_offset = keys.since as usize;
                let since_offset = keys.since as usize;
//                let until_offset = keys.until as usize;
                let until_offset = keys.until as usize;
//                debug_assert!(since_offset <= until_offset);
                debug_assert!(since_offset <= until_offset);
//                let len = until_offset - since_offset;
                let len = until_offset - since_offset;
//                match len {
                match len {
//                    0 => &[],
                    0 => &[],
//                    _ => std::slice::from_raw_parts(keys.ptr.add(since_offset), len),
                    _ => std::slice::from_raw_parts(keys.ptr.add(since_offset), len),
//                }
                }
//            }
            }
//            .iter()
            .iter()
//            .map(|x| {
            .map(|x| {
//                Felt::from_bytes_be(&{
                Felt::from_bytes_be(&{
//                    let mut data = x.0;
                    let mut data = x.0;
//                    data.reverse();
                    data.reverse();
//                    data
                    data
//                })
                })
//            })
            })
//            .collect();
            .collect();
//

//            let data: Vec<_> = unsafe {
            let data: Vec<_> = unsafe {
//                let since_offset = data.since as usize;
                let since_offset = data.since as usize;
//                let until_offset = data.until as usize;
                let until_offset = data.until as usize;
//                debug_assert!(since_offset <= until_offset);
                debug_assert!(since_offset <= until_offset);
//                let len = until_offset - since_offset;
                let len = until_offset - since_offset;
//                match len {
                match len {
//                    0 => &[],
                    0 => &[],
//                    _ => std::slice::from_raw_parts(data.ptr.add(since_offset), len),
                    _ => std::slice::from_raw_parts(data.ptr.add(since_offset), len),
//                }
                }
//            }
            }
//            .iter()
            .iter()
//            .map(|x| {
            .map(|x| {
//                Felt::from_bytes_be(&{
                Felt::from_bytes_be(&{
//                    let mut data = x.0;
                    let mut data = x.0;
//                    data.reverse();
                    data.reverse();
//                    data
                    data
//                })
                })
//            })
            })
//            .collect();
            .collect();
//

//            let result = ptr.emit_event(&keys, &data, gas);
            let result = ptr.emit_event(&keys, &data, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(_) => SyscallResultAbi {
                Ok(_) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(()),
                        payload: ManuallyDrop::new(()),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_send_message_to_l1(
        extern "C" fn wrap_send_message_to_l1(
//            result_ptr: &mut SyscallResultAbi<()>,
            result_ptr: &mut SyscallResultAbi<()>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            to_address: &Felt252Abi,
            to_address: &Felt252Abi,
//            payload: &ArrayAbi<Felt252Abi>,
            payload: &ArrayAbi<Felt252Abi>,
//        ) {
        ) {
//            let to_address = Felt::from_bytes_be(&{
            let to_address = Felt::from_bytes_be(&{
//                let mut data = to_address.0;
                let mut data = to_address.0;
//                data.reverse();
                data.reverse();
//                data
                data
//            });
            });
//            let payload: Vec<_> = unsafe {
            let payload: Vec<_> = unsafe {
//                let since_offset = payload.since as usize;
                let since_offset = payload.since as usize;
//                let until_offset = payload.until as usize;
                let until_offset = payload.until as usize;
//                debug_assert!(since_offset <= until_offset);
                debug_assert!(since_offset <= until_offset);
//                let len = until_offset - since_offset;
                let len = until_offset - since_offset;
//                match len {
                match len {
//                    0 => &[],
                    0 => &[],
//                    _ => std::slice::from_raw_parts(payload.ptr.add(since_offset), len),
                    _ => std::slice::from_raw_parts(payload.ptr.add(since_offset), len),
//                }
                }
//            }
            }
//            .iter()
            .iter()
//            .map(|x| {
            .map(|x| {
//                Felt::from_bytes_be(&{
                Felt::from_bytes_be(&{
//                    let mut data = x.0;
                    let mut data = x.0;
//                    data.reverse();
                    data.reverse();
//                    data
                    data
//                })
                })
//            })
            })
//            .collect();
            .collect();
//

//            let result = ptr.send_message_to_l1(to_address, &payload, gas);
            let result = ptr.send_message_to_l1(to_address, &payload, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(_) => SyscallResultAbi {
                Ok(_) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(()),
                        payload: ManuallyDrop::new(()),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_keccak(
        extern "C" fn wrap_keccak(
//            result_ptr: &mut SyscallResultAbi<U256>,
            result_ptr: &mut SyscallResultAbi<U256>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            input: &ArrayAbi<u64>,
            input: &ArrayAbi<u64>,
//        ) {
        ) {
//            let input = unsafe {
            let input = unsafe {
//                let since_offset = input.since as usize;
                let since_offset = input.since as usize;
//                let until_offset = input.until as usize;
                let until_offset = input.until as usize;
//                debug_assert!(since_offset <= until_offset);
                debug_assert!(since_offset <= until_offset);
//                let len = until_offset - since_offset;
                let len = until_offset - since_offset;
//                match len {
                match len {
//                    0 => &[],
                    0 => &[],
//                    _ => std::slice::from_raw_parts(input.ptr.add(since_offset), len),
                    _ => std::slice::from_raw_parts(input.ptr.add(since_offset), len),
//                }
                }
//            };
            };
//

//            let result = ptr.keccak(input, gas);
            let result = ptr.keccak(input, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(x),
                        payload: ManuallyDrop::new(x),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_secp256k1_new(
        extern "C" fn wrap_secp256k1_new(
//            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256k1Point>)>,
            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256k1Point>)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            x: &U256,
            x: &U256,
//            y: &U256,
            y: &U256,
//        ) {
        ) {
//            let result = ptr.secp256k1_new(*x, *y, gas);
            let result = ptr.secp256k1_new(*x, *y, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(match x {
                        payload: ManuallyDrop::new(match x {
//                            Some(x) => (0, MaybeUninit::new(x)),
                            Some(x) => (0, MaybeUninit::new(x)),
//                            None => (1, MaybeUninit::uninit()),
                            None => (1, MaybeUninit::uninit()),
//                        }),
                        }),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_secp256k1_add(
        extern "C" fn wrap_secp256k1_add(
//            result_ptr: &mut SyscallResultAbi<Secp256k1Point>,
            result_ptr: &mut SyscallResultAbi<Secp256k1Point>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p0: &Secp256k1Point,
            p0: &Secp256k1Point,
//            p1: &Secp256k1Point,
            p1: &Secp256k1Point,
//        ) {
        ) {
//            let result = ptr.secp256k1_add(*p0, *p1, gas);
            let result = ptr.secp256k1_add(*p0, *p1, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(x),
                        payload: ManuallyDrop::new(x),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_secp256k1_mul(
        extern "C" fn wrap_secp256k1_mul(
//            result_ptr: &mut SyscallResultAbi<Secp256k1Point>,
            result_ptr: &mut SyscallResultAbi<Secp256k1Point>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p: &Secp256k1Point,
            p: &Secp256k1Point,
//            scalar: &U256,
            scalar: &U256,
//        ) {
        ) {
//            let result = ptr.secp256k1_mul(*p, *scalar, gas);
            let result = ptr.secp256k1_mul(*p, *scalar, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(x),
                        payload: ManuallyDrop::new(x),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_secp256k1_get_point_from_x(
        extern "C" fn wrap_secp256k1_get_point_from_x(
//            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256k1Point>)>,
            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256k1Point>)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            x: &U256,
            x: &U256,
//            y_parity: &bool,
            y_parity: &bool,
//        ) {
        ) {
//            let result = ptr.secp256k1_get_point_from_x(*x, *y_parity, gas);
            let result = ptr.secp256k1_get_point_from_x(*x, *y_parity, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(match x {
                        payload: ManuallyDrop::new(match x {
//                            Some(x) => (0, MaybeUninit::new(x)),
                            Some(x) => (0, MaybeUninit::new(x)),
//                            None => (1, MaybeUninit::uninit()),
                            None => (1, MaybeUninit::uninit()),
//                        }),
                        }),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_secp256k1_get_xy(
        extern "C" fn wrap_secp256k1_get_xy(
//            result_ptr: &mut SyscallResultAbi<(U256, U256)>,
            result_ptr: &mut SyscallResultAbi<(U256, U256)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p: &Secp256k1Point,
            p: &Secp256k1Point,
//        ) {
        ) {
//            let result = ptr.secp256k1_get_xy(*p, gas);
            let result = ptr.secp256k1_get_xy(*p, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(x),
                        payload: ManuallyDrop::new(x),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_secp256r1_new(
        extern "C" fn wrap_secp256r1_new(
//            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256r1Point>)>,
            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256r1Point>)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            x: &U256,
            x: &U256,
//            y: &U256,
            y: &U256,
//        ) {
        ) {
//            let result = ptr.secp256r1_new(*x, *y, gas);
            let result = ptr.secp256r1_new(*x, *y, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(match x {
                        payload: ManuallyDrop::new(match x {
//                            Some(x) => (0, MaybeUninit::new(x)),
                            Some(x) => (0, MaybeUninit::new(x)),
//                            None => (1, MaybeUninit::uninit()),
                            None => (1, MaybeUninit::uninit()),
//                        }),
                        }),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_secp256r1_add(
        extern "C" fn wrap_secp256r1_add(
//            result_ptr: &mut SyscallResultAbi<Secp256r1Point>,
            result_ptr: &mut SyscallResultAbi<Secp256r1Point>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p0: &Secp256r1Point,
            p0: &Secp256r1Point,
//            p1: &Secp256r1Point,
            p1: &Secp256r1Point,
//        ) {
        ) {
//            let result = ptr.secp256r1_add(*p0, *p1, gas);
            let result = ptr.secp256r1_add(*p0, *p1, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(x),
                        payload: ManuallyDrop::new(x),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_secp256r1_mul(
        extern "C" fn wrap_secp256r1_mul(
//            result_ptr: &mut SyscallResultAbi<Secp256r1Point>,
            result_ptr: &mut SyscallResultAbi<Secp256r1Point>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p: &Secp256r1Point,
            p: &Secp256r1Point,
//            scalar: &U256,
            scalar: &U256,
//        ) {
        ) {
//            let result = ptr.secp256r1_mul(*p, *scalar, gas);
            let result = ptr.secp256r1_mul(*p, *scalar, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(x),
                        payload: ManuallyDrop::new(x),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_secp256r1_get_point_from_x(
        extern "C" fn wrap_secp256r1_get_point_from_x(
//            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256r1Point>)>,
            result_ptr: &mut SyscallResultAbi<(u8, MaybeUninit<Secp256r1Point>)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            x: &U256,
            x: &U256,
//            y_parity: &bool,
            y_parity: &bool,
//        ) {
        ) {
//            let result = ptr.secp256r1_get_point_from_x(*x, *y_parity, gas);
            let result = ptr.secp256r1_get_point_from_x(*x, *y_parity, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(match x {
                        payload: ManuallyDrop::new(match x {
//                            Some(x) => (0, MaybeUninit::new(x)),
                            Some(x) => (0, MaybeUninit::new(x)),
//                            None => (1, MaybeUninit::uninit()),
                            None => (1, MaybeUninit::uninit()),
//                        }),
                        }),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//

//        extern "C" fn wrap_secp256r1_get_xy(
        extern "C" fn wrap_secp256r1_get_xy(
//            result_ptr: &mut SyscallResultAbi<(U256, U256)>,
            result_ptr: &mut SyscallResultAbi<(U256, U256)>,
//            ptr: &mut T,
            ptr: &mut T,
//            gas: &mut u128,
            gas: &mut u128,
//            p: &Secp256r1Point,
            p: &Secp256r1Point,
//        ) {
        ) {
//            let result = ptr.secp256r1_get_xy(*p, gas);
            let result = ptr.secp256r1_get_xy(*p, gas);
//

//            *result_ptr = match result {
            *result_ptr = match result {
//                Ok(x) => SyscallResultAbi {
                Ok(x) => SyscallResultAbi {
//                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
//                        tag: 0u8,
                        tag: 0u8,
//                        payload: ManuallyDrop::new(x),
                        payload: ManuallyDrop::new(x),
//                    }),
                    }),
//                },
                },
//                Err(e) => Self::wrap_error(&e),
                Err(e) => Self::wrap_error(&e),
//            };
            };
//        }
        }
//    }
    }
//}
}
