use cairo_felt::Felt252;

type SyscallResult<T> = std::result::Result<T, Vec<Felt252>>;

/// Binary representation of a `felt252` (in MLIR).
#[repr(C, align(8))]
struct Felt252Abi(pub [u8; 32]);
/// Binary representation of a `u256` (in MLIR).
// TODO: This shouldn't need to be public.
#[repr(C, align(8))]
pub struct U256(pub [u8; 32]);

pub struct ExecutionInfo {
    // TODO: Add fields.
}
pub struct Secp256k1Point {
    // TODO: Add fields.
}
pub struct Secp256r1Point {
    // TODO: Add fields.
}

pub trait StarkNetSyscallHandler {
    fn get_block_hash(&self, block_number: u64) -> SyscallResult<Felt252>;
    fn get_execution_info(&self) -> SyscallResult<ExecutionInfo>;

    fn deploy(
        &self,
        class_hash: Felt252,
        contract_address_salt: Felt252,
        calldata: &[Felt252],
        deploy_from_zero: bool,
    ) -> SyscallResult<(Felt252, Vec<Felt252>)>;
    fn replace_class(&self, class_hash: Felt252) -> SyscallResult<()>;

    fn library_call(
        &self,
        class_hash: Felt252,
        function_selector: Felt252,
        calldata: &[Felt252],
    ) -> SyscallResult<Vec<Felt252>>;
    fn call_contract(
        &self,
        address: Felt252,
        entry_point_selector: Felt252,
        calldata: &[Felt252],
    ) -> SyscallResult<Vec<Felt252>>;

    fn storage_read(&self, address_domain: u32, address: Felt252) -> SyscallResult<Felt252>;
    fn storage_write(
        &self,
        address_domain: u32,
        address: Felt252,
        value: Felt252,
    ) -> SyscallResult<()>;

    fn emit_event(&self, keys: &[Felt252], data: &[Felt252]) -> SyscallResult<()>;
    fn send_message_to_l1(&self, to_address: Felt252, payload: &[Felt252]) -> SyscallResult<()>;

    fn keccak(&self, input: &[u64]) -> SyscallResult<U256>;

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

    // Testing syscalls.
    // TODO: Make them optional. Crash if called but not implemented.
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

// TODO: Move to the correct place or remove if unused.
mod handler {
    use super::*;
    use std::{alloc::Layout, mem::ManuallyDrop, ptr::NonNull};

    #[repr(C)]
    struct SyscallResultAbi<T> {
        tag: u8,
        payload: SyscallResultPayloadAbi<T>,
    }

    #[repr(C)]
    union SyscallResultPayloadAbi<T> {
        ok: ManuallyDrop<T>,
        err: (NonNull<Felt252Abi>, u32, u32),
    }

    #[repr(C)]
    struct StarkNetSyscallHandlerCallbacks<'a, T>
    where
        T: StarkNetSyscallHandler,
    {
        self_ptr: &'a T,

        get_block_hash: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            ptr: &mut T,
            gas: &mut u64,
            block_number: u64,
        ),
        get_execution_info: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<ExecutionInfo>,
            ptr: &mut T,
            gas: &mut u64,
        ),
        deploy: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<(Felt252Abi, Vec<Felt252Abi>)>,
            ptr: &mut T,
            gas: &mut u64,
            class_hash: &Felt252Abi,
            contract_address_salt: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
            deploy_from_zero: bool,
        ),
    }

    impl<'a, T> StarkNetSyscallHandlerCallbacks<'a, T>
    where
        T: StarkNetSyscallHandler,
    {
        pub fn new(handler: &'a T) -> Self {
            Self {
                self_ptr: handler,
                get_block_hash: Self::wrap_get_block_hash,
                get_execution_info: Self::get_execution_info,
                deploy: Self::deploy,
            }
        }

        extern "C" fn wrap_get_block_hash(
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            ptr: &mut T,
            _gas: &mut u64,
            block_number: u64,
        ) {
            // TODO: Handle gas.
            let result = ptr.get_block_hash(block_number);

            *result_ptr = match result {
                Ok(x) => SyscallResultAbi {
                    tag: 0u8,
                    payload: SyscallResultPayloadAbi {
                        ok: ManuallyDrop::new(Felt252Abi(x.to_le_bytes())),
                    },
                },
                Err(e) => SyscallResultAbi {
                    tag: 1u8,
                    payload: unsafe {
                        let ptr = libc::malloc(Layout::array::<Felt252Abi>(e.len()).unwrap().size())
                            as *mut Felt252Abi;

                        let len: u32 = e.len().try_into().unwrap();
                        for (i, val) in e.into_iter().enumerate() {
                            ptr.add(i).write(Felt252Abi(val.to_le_bytes()));
                        }

                        SyscallResultPayloadAbi {
                            err: (NonNull::new(ptr).unwrap(), len, len),
                        }
                    },
                },
            };
        }

        extern "C" fn get_execution_info(
            result_ptr: &mut SyscallResultAbi<ExecutionInfo>,
            ptr: &mut T,
            _gas: &mut u64,
        ) {
            // TODO: handle gas
            let result = ptr.get_execution_info();

            *result_ptr = match result {
                Ok(x) => SyscallResultAbi {
                    tag: 0u8,
                    payload: SyscallResultPayloadAbi {
                        ok: ManuallyDrop::new(x),
                    },
                },
                Err(e) => SyscallResultAbi {
                    tag: 1u8,
                    payload: unsafe {
                        let ptr = libc::malloc(Layout::array::<Felt252Abi>(e.len()).unwrap().size())
                            as *mut Felt252Abi;

                        let len: u32 = e.len().try_into().unwrap();
                        for (i, val) in e.into_iter().enumerate() {
                            ptr.add(i).write(Felt252Abi(val.to_le_bytes()));
                        }

                        SyscallResultPayloadAbi {
                            err: (NonNull::new(ptr).unwrap(), len, len),
                        }
                    },
                },
            };
        }

        // TODO: change all from_bytes_be to from_bytes_ne when added.

        extern "C" fn deploy(
            result_ptr: &mut SyscallResultAbi<(Felt252Abi, Vec<Felt252Abi>)>,
            ptr: &mut T,
            _gas: &mut u64,
            class_hash: &Felt252Abi,
            contract_address_salt: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
            deploy_from_zero: bool,
        ) {
            // TODO: handle gas
            let class_hash = Felt252::from_bytes_be(&class_hash.0);
            let contract_address_salt = Felt252::from_bytes_be(&contract_address_salt.0);

            let calldata: Vec<_> = unsafe {
                let len = (*calldata).1 as usize;

                std::slice::from_raw_parts((*calldata).0, len)
            }
            .iter()
            .map(|x| Felt252::from_bytes_be(&x.0))
            .collect();

            let result = ptr.deploy(
                class_hash,
                contract_address_salt,
                &calldata,
                deploy_from_zero,
            );

            *result_ptr = match result {
                Ok(x) => {
                    let felts: Vec<_> = x.1.iter().map(|x| Felt252Abi(x.to_le_bytes())).collect();
                    SyscallResultAbi {
                        tag: 0u8,
                        payload: SyscallResultPayloadAbi {
                            ok: ManuallyDrop::new((Felt252Abi(x.0.to_le_bytes()), felts)),
                        },
                    }
                }
                Err(e) => SyscallResultAbi {
                    tag: 1u8,
                    payload: unsafe {
                        let ptr = libc::malloc(Layout::array::<Felt252Abi>(e.len()).unwrap().size())
                            as *mut Felt252Abi;

                        let len: u32 = e.len().try_into().unwrap();
                        for (i, val) in e.into_iter().enumerate() {
                            ptr.add(i).write(Felt252Abi(val.to_le_bytes()));
                        }

                        SyscallResultPayloadAbi {
                            err: (NonNull::new(ptr).unwrap(), len, len),
                        }
                    },
                },
            };
        }
    }
}
