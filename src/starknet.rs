type SyscallResult<T> = std::result::Result<T, std::convert::Infallible>;

type Felt252 = [u8; 32];
type U256 = [u8; 32];

struct ExecutionInfo {}
struct Secp256k1Point {}
struct Secp256r1Point {}

trait StarkNetSyscallHandler {
    fn get_block_hash(&self, block_number: u64) -> Felt252;
    fn get_execution_info(&self) -> ExecutionInfo;

    fn deploy(
        &self,
        class_hash: Felt252,
        contract_address_salt: Felt252,
        calldata: &[Felt252],
        deploy_from_zero: bool,
    ) -> SyscallResult<(Felt252, &[Felt252])>;
    fn replace_class(&self, class_hash: Felt252) -> SyscallResult<()>;

    fn library_call(
        &self,
        class_hash: Felt252,
        function_selector: Felt252,
        calldata: &[Felt252],
    ) -> &[Felt252];
    fn call_contract(
        &self,
        address: Felt252,
        entry_point_selector: Felt252,
        calldata: &[Felt252],
    ) -> &[Felt252];

    fn storage_read(&self, address_domain: u32, address: Felt252) -> SyscallResult<Felt252>;
    fn storage_write(
        &self,
        address_domain: u32,
        address: Felt252,
        value: Felt252,
    ) -> SyscallResult<()>;

    fn emit_event(&self, keys: &[Felt252], data: &[Felt252]) -> SyscallResult<()>;
    fn send_message_to_l1(&self, to_address: Felt252, payload: &[Felt252]);

    fn keccak(&self, input: &[u64]) -> SyscallResult<()>;

    // TODO: secp256k1 syscalls
    fn secp256k1_add(
        &self,
        p0: Secp256k1Point,
        p1: Secp256k1Point,
    ) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256k1_get_point_from_x(
        &self,
        x: U256,
        y_parity: bool,
    ) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256k1_get_xy(&self, p: Secp256k1Point) -> SyscallResult<(U256, U256)>;
    fn secp256k1_mul(&self, p: Secp256k1Point, m: U256) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256k1_new(&self, x: U256, y: U256) -> SyscallResult<Option<Secp256k1Point>>;

    // TODO: secp256r1 syscalls
    fn secp256r1_add(
        &self,
        p0: Secp256k1Point,
        p1: Secp256k1Point,
    ) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256r1_get_point_from_x(
        &self,
        x: U256,
        y_parity: bool,
    ) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256r1_get_xy(&self, p: Secp256k1Point) -> SyscallResult<(U256, U256)>;
    fn secp256r1_mul(&self, p: Secp256k1Point, m: U256) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256r1_new(&self, x: U256, y: U256) -> SyscallResult<Option<Secp256k1Point>>;

    // TODO: Testing syscalls
    fn pop_log(&self);
    fn set_account_contract_address(&self, contract_address: Felt252);
    fn set_block_number(&self, block_number: u64);
    fn set_block_timestamp(&self, block_timestamp: u64);
    fn set_caller_address(&self, address: Felt252);
    fn set_chain_id(&self, chain_id: Felt252);
    fn set_contract_address(&self, address: Felt252);
    fn set_max_fee(&self, max_fee: u128);
    fn set_nonce(&self, nonce: Felt252);
    fn set_sequencer_address(&self, address: Felt252);
    fn set_signature(&self, signature: &[Felt252]);
    fn set_transaction_hash(&self, transaction_hash: Felt252);
    fn set_version(&self, version: Felt252);
}

mod handler {
    use super::*;

    #[repr(C)]
    struct StarkNetSyscallHandlerCallbacks<'a, T>
    where
        T: StarkNetSyscallHandler,
    {
        self_ptr: &'a T,

        get_block_hash: extern "C" fn(ptr: &mut T, block_number: u64),
    }

    impl<'a, T> StarkNetSyscallHandlerCallbacks<'a, T>
    where
        T: StarkNetSyscallHandler,
    {
        pub fn new(handler: &'a T) -> Self {
            Self {
                self_ptr: handler,
                get_block_hash: Self::wrap_get_block_hash,
            }
        }

        extern "C" fn wrap_get_block_hash(ptr: &mut T, block_number: u64) {
            // TODO: Handle result.
            ptr.get_block_hash(block_number);
        }
    }
}
