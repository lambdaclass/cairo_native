//! Starknet related code for `cairo_native`

#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use starknet_types_core::felt::Felt;

pub type SyscallResult<T> = std::result::Result<T, Vec<Felt>>;

/// Binary representation of a `Felt` (in MLIR).
#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "x86_64", repr(C, align(8)))]
#[cfg_attr(not(target_arch = "x86_64"), repr(C, align(16)))]
struct Felt252Abi(pub [u8; 32]);
/// Binary representation of a `u256` (in MLIR).
// TODO: This shouldn't need to be public.
#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "x86_64", repr(C, align(8)))]
#[cfg_attr(not(target_arch = "x86_64"), repr(C, align(16)))]
pub struct U256(pub [u8; 32]);

pub struct ExecutionInfo {
    pub block_info: BlockInfo,
    pub tx_info: TxInfo,
    pub caller_address: Felt,
    pub contract_address: Felt,
    pub entry_point_selector: Felt,
}

pub struct BlockInfo {
    pub block_number: u64,
    pub block_timestamp: u64,
    pub sequencer_address: Felt,
}

pub struct TxInfo {
    pub version: Felt,
    pub account_contract_address: Felt,
    pub max_fee: u128,
    pub signature: Vec<Felt>,
    pub transaction_hash: Felt,
    pub chain_id: Felt,
    pub nonce: Felt,
}

pub struct Secp256k1Point {
    // TODO: Add fields.
}

pub struct Secp256r1Point {
    // TODO: Add fields.
}

pub trait StarkNetSyscallHandler {
    fn get_block_hash(
        &mut self,
        block_number: u64,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Felt>;
    fn get_execution_info(&mut self, remaining_gas: &mut u128) -> SyscallResult<ExecutionInfo>;

    fn deploy(
        &mut self,
        class_hash: Felt,
        contract_address_salt: Felt,
        calldata: &[Felt],
        deploy_from_zero: bool,
        remaining_gas: &mut u128,
    ) -> SyscallResult<(Felt, Vec<Felt>)>;
    fn replace_class(&mut self, class_hash: Felt, remaining_gas: &mut u128) -> SyscallResult<()>;

    fn library_call(
        &mut self,
        class_hash: Felt,
        function_selector: Felt,
        calldata: &[Felt],
        remaining_gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>>;

    fn call_contract(
        &mut self,
        address: Felt,
        entry_point_selector: Felt,
        calldata: &[Felt],
        remaining_gas: &mut u128,
    ) -> SyscallResult<Vec<Felt>>;

    fn storage_read(
        &mut self,
        address_domain: u32,
        address: Felt,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Felt>;

    fn storage_write(
        &mut self,
        address_domain: u32,
        address: Felt,
        value: Felt,
        remaining_gas: &mut u128,
    ) -> SyscallResult<()>;

    fn emit_event(
        &mut self,
        keys: &[Felt],
        data: &[Felt],
        remaining_gas: &mut u128,
    ) -> SyscallResult<()>;

    fn send_message_to_l1(
        &mut self,
        to_address: Felt,
        payload: &[Felt],
        remaining_gas: &mut u128,
    ) -> SyscallResult<()>;

    fn keccak(&mut self, input: &[u64], remaining_gas: &mut u128) -> SyscallResult<U256>;

    // TODO: secp256k1 syscalls
    fn secp256k1_add(
        &mut self,
        p0: Secp256k1Point,
        p1: Secp256k1Point,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>>;

    fn secp256k1_get_point_from_x(
        &self,
        x: U256,
        y_parity: bool,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>>;

    fn secp256k1_get_xy(
        &self,
        p: Secp256k1Point,
        remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)>;

    fn secp256k1_mul(
        &self,
        p: Secp256k1Point,
        m: U256,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>>;

    fn secp256k1_new(
        &self,
        x: U256,
        y: U256,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>>;

    // TODO: secp256r1 syscalls
    fn secp256r1_add(
        &self,
        p0: Secp256k1Point,
        p1: Secp256k1Point,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>>;

    fn secp256r1_get_point_from_x(
        &self,
        x: U256,
        y_parity: bool,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>>;

    fn secp256r1_get_xy(
        &self,
        p: Secp256k1Point,
        remaining_gas: &mut u128,
    ) -> SyscallResult<(U256, U256)>;

    fn secp256r1_mul(
        &self,
        p: Secp256k1Point,
        m: U256,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>>;

    fn secp256r1_new(
        &mut self,
        x: U256,
        y: U256,
        remaining_gas: &mut u128,
    ) -> SyscallResult<Option<Secp256k1Point>>;

    // Testing syscalls.
    // TODO: Make them optional. Crash if called but not implemented.
    fn pop_log(&mut self);

    fn set_account_contract_address(&mut self, contract_address: Felt);

    fn set_block_number(&mut self, block_number: u64);

    fn set_block_timestamp(&mut self, block_timestamp: u64);

    fn set_caller_address(&mut self, address: Felt);

    fn set_chain_id(&mut self, chain_id: Felt);

    fn set_contract_address(&mut self, address: Felt);

    fn set_max_fee(&mut self, max_fee: u128);

    fn set_nonce(&mut self, nonce: Felt);

    fn set_sequencer_address(&mut self, address: Felt);

    fn set_signature(&mut self, signature: &[Felt]);

    fn set_transaction_hash(&mut self, transaction_hash: Felt);

    fn set_version(&mut self, version: Felt);
}

// TODO: Move to the correct place or remove if unused.
pub(crate) mod handler {
    use super::*;
    use std::{
        alloc::Layout,
        fmt::Debug,
        mem::{size_of, ManuallyDrop},
        ptr::NonNull,
    };

    macro_rules! field_offset {
        ( $ident:path, $field:ident ) => {
            unsafe {
                let value_ptr = std::mem::MaybeUninit::<$ident>::uninit().as_ptr();
                let field_ptr: *const u8 = std::ptr::addr_of!((*value_ptr).$field) as *const u8;
                field_ptr.offset_from(value_ptr as *const u8) as usize
            }
        };
    }

    #[repr(C)]
    union SyscallResultAbi<T> {
        ok: ManuallyDrop<SyscallResultAbiOk<T>>,
        err: ManuallyDrop<SyscallResultAbiErr>,
    }

    #[repr(C)]
    struct SyscallResultAbiOk<T> {
        tag: u8,
        payload: ManuallyDrop<T>,
    }

    #[repr(C)]
    struct SyscallResultAbiErr {
        tag: u8,
        payload: (NonNull<Felt252Abi>, u32, u32),
    }

    #[repr(C)]
    struct ExecutionInfoAbi {
        block_info: NonNull<BlockInfoAbi>,
        tx_info: NonNull<TxInfoAbi>,
        caller_address: Felt252Abi,
        contract_address: Felt252Abi,
        entry_point_selector: Felt252Abi,
    }

    #[repr(C)]
    struct BlockInfoAbi {
        block_number: u64,
        block_timestamp: u64,
        sequencer_address: Felt252Abi,
    }

    #[repr(C)]
    struct TxInfoAbi {
        version: Felt252Abi,
        account_contract_address: Felt252Abi,
        max_fee: u128,
        signature: (NonNull<Felt252Abi>, u32, u32),
        transaction_hash: Felt252Abi,
        chain_id: Felt252Abi,
        nonce: Felt252Abi,
    }

    #[repr(C)]
    #[derive(Debug)]
    pub struct StarkNetSyscallHandlerCallbacks<'a, T> {
        self_ptr: &'a mut T,

        get_block_hash: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            ptr: &mut T,
            gas: &mut u128,
            block_number: u64,
        ),
        get_execution_info: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<NonNull<ExecutionInfoAbi>>,
            ptr: &mut T,
            gas: &mut u128,
        ),
        deploy: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<(Felt252Abi, (NonNull<Felt252Abi>, u32, u32))>,
            ptr: &mut T,
            gas: &mut u128,
            class_hash: &Felt252Abi,
            contract_address_salt: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
            deploy_from_zero: bool,
        ),
        replace_class: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            _gas: &mut u128,
            class_hash: &Felt252Abi,
        ),
        library_call: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<(NonNull<Felt252Abi>, u32, u32)>,
            ptr: &mut T,
            gas: &mut u128,
            class_hash: &Felt252Abi,
            function_selector: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
        ),
        call_contract: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<(NonNull<Felt252Abi>, u32, u32)>,
            ptr: &mut T,
            gas: &mut u128,
            address: &Felt252Abi,
            entry_point_selector: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
        ),

        storage_read: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            ptr: &mut T,
            gas: &mut u128,
            address_domain: u32,
            address: &Felt252Abi,
        ),
        storage_write: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            gas: &mut u128,
            address_domain: u32,
            address: &Felt252Abi,
            value: &Felt252Abi,
        ),
        emit_event: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            gas: &mut u128,
            keys: *const (*const Felt252Abi, u32, u32),
            data: *const (*const Felt252Abi, u32, u32),
        ),
        send_message_to_l1: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            gas: &mut u128,
            to_address: &Felt252Abi,
            data: *const (*const Felt252Abi, u32, u32),
        ),
        keccak: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<U256>,
            ptr: &mut T,
            _gas: &mut u128,
            input: *const (*const u64, u32, u32),
        ),
    }

    impl<'a, T> StarkNetSyscallHandlerCallbacks<'a, T>
    where
        T: 'a,
    {
        // Callback field indices.
        pub const CALL_CONTRACT: usize = field_offset!(Self, call_contract) >> 3;
        pub const DEPLOY: usize = field_offset!(Self, deploy) >> 3;
        pub const EMIT_EVENT: usize = field_offset!(Self, emit_event) >> 3;
        pub const GET_BLOCK_HASH: usize = field_offset!(Self, get_block_hash) >> 3;
        pub const GET_EXECUTION_INFO: usize = field_offset!(Self, get_execution_info) >> 3;
        pub const KECCAK: usize = field_offset!(Self, keccak) >> 3;
        pub const LIBRARY_CALL: usize = field_offset!(Self, library_call) >> 3;
        pub const REPLACE_CLASS: usize = field_offset!(Self, replace_class) >> 3;
        pub const SEND_MESSAGE_TO_L1: usize = field_offset!(Self, send_message_to_l1) >> 3;
        pub const STORAGE_READ: usize = field_offset!(Self, storage_read) >> 3;
        pub const STORAGE_WRITE: usize = field_offset!(Self, storage_write) >> 3;
    }

    impl<'a, T> StarkNetSyscallHandlerCallbacks<'a, T>
    where
        T: StarkNetSyscallHandler + 'a,
    {
        pub fn new(handler: &'a mut T) -> Self {
            Self {
                self_ptr: handler,
                get_block_hash: Self::wrap_get_block_hash,
                get_execution_info: Self::wrap_get_execution_info,
                deploy: Self::wrap_deploy,
                replace_class: Self::wrap_replace_class,
                library_call: Self::wrap_library_call,
                call_contract: Self::wrap_call_contract,
                storage_read: Self::wrap_storage_read,
                storage_write: Self::wrap_storage_write,
                emit_event: Self::wrap_emit_event,
                send_message_to_l1: Self::wrap_send_message_to_l1,
                keccak: Self::wrap_keccak,
            }
        }

        unsafe fn alloc_mlir_array<E: Clone>(data: &[E]) -> (NonNull<E>, u32, u32) {
            let ptr = libc::malloc(Layout::array::<E>(data.len()).unwrap().size()) as *mut E;

            let len: u32 = data.len().try_into().unwrap();
            for (i, val) in data.iter().enumerate() {
                ptr.add(i).write(val.clone());
            }

            (NonNull::new(ptr).unwrap(), len, len)
        }

        fn wrap_error<E>(e: &[Felt]) -> SyscallResultAbi<E> {
            SyscallResultAbi {
                err: ManuallyDrop::new(SyscallResultAbiErr {
                    tag: 1u8,
                    payload: unsafe {
                        let data: Vec<_> = e.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
                        Self::alloc_mlir_array(&data)
                    },
                }),
            }
        }

        extern "C" fn wrap_get_block_hash(
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            ptr: &mut T,
            gas: &mut u128,
            block_number: u64,
        ) {
            let result = ptr.get_block_hash(block_number, gas);

            *result_ptr = match result {
                Ok(x) => SyscallResultAbi {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                        tag: 0u8,
                        payload: ManuallyDrop::new(Felt252Abi(x.to_bytes_le())),
                    }),
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_get_execution_info(
            result_ptr: &mut SyscallResultAbi<NonNull<ExecutionInfoAbi>>,
            ptr: &mut T,
            gas: &mut u128,
        ) {
            let result = ptr.get_execution_info(gas);

            *result_ptr = match result {
                Ok(x) => SyscallResultAbi {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                        tag: 0u8,
                        payload: unsafe {
                            let mut block_info_ptr =
                                NonNull::new(
                                    libc::malloc(size_of::<BlockInfoAbi>()) as *mut BlockInfoAbi
                                )
                                .unwrap();
                            block_info_ptr.as_mut().block_number = x.block_info.block_number;
                            block_info_ptr.as_mut().block_timestamp = x.block_info.block_timestamp;
                            block_info_ptr.as_mut().sequencer_address =
                                Felt252Abi(x.block_info.sequencer_address.to_bytes_le());

                            let mut tx_info_ptr = NonNull::new(
                                libc::malloc(size_of::<TxInfoAbi>()) as *mut TxInfoAbi,
                            )
                            .unwrap();
                            tx_info_ptr.as_mut().version =
                                Felt252Abi(x.tx_info.version.to_bytes_le());
                            tx_info_ptr.as_mut().account_contract_address =
                                Felt252Abi(x.tx_info.account_contract_address.to_bytes_le());
                            tx_info_ptr.as_mut().max_fee = x.tx_info.max_fee;
                            tx_info_ptr.as_mut().signature = Self::alloc_mlir_array(
                                &x.tx_info
                                    .signature
                                    .into_iter()
                                    .map(|x| Felt252Abi(x.to_bytes_le()))
                                    .collect::<Vec<_>>(),
                            );
                            tx_info_ptr.as_mut().transaction_hash =
                                Felt252Abi(x.tx_info.transaction_hash.to_bytes_le());
                            tx_info_ptr.as_mut().chain_id =
                                Felt252Abi(x.tx_info.chain_id.to_bytes_le());
                            tx_info_ptr.as_mut().nonce = Felt252Abi(x.tx_info.nonce.to_bytes_le());

                            let mut execution_info_ptr =
                                NonNull::new(libc::malloc(size_of::<ExecutionInfoAbi>())
                                    as *mut ExecutionInfoAbi)
                                .unwrap();
                            execution_info_ptr.as_mut().block_info = block_info_ptr;
                            execution_info_ptr.as_mut().tx_info = tx_info_ptr;
                            execution_info_ptr.as_mut().caller_address =
                                Felt252Abi(x.caller_address.to_bytes_le());
                            execution_info_ptr.as_mut().contract_address =
                                Felt252Abi(x.contract_address.to_bytes_le());
                            execution_info_ptr.as_mut().entry_point_selector =
                                Felt252Abi(x.entry_point_selector.to_bytes_le());

                            ManuallyDrop::new(execution_info_ptr)
                        },
                    }),
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        // TODO: change all from_bytes_be to from_bytes_ne when added and undo byte swapping.

        extern "C" fn wrap_deploy(
            result_ptr: &mut SyscallResultAbi<(Felt252Abi, (NonNull<Felt252Abi>, u32, u32))>,
            ptr: &mut T,
            gas: &mut u128,
            class_hash: &Felt252Abi,
            contract_address_salt: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
            deploy_from_zero: bool,
        ) {
            let class_hash = Felt::from_bytes_be(&{
                let mut data = class_hash.0;
                data.reverse();
                data
            });
            let contract_address_salt = Felt::from_bytes_be(&{
                let mut data = contract_address_salt.0;
                data.reverse();
                data
            });

            let calldata: Vec<_> = unsafe {
                let len = (*calldata).1 as usize;
                std::slice::from_raw_parts((*calldata).0, len)
            }
            .iter()
            .map(|x| {
                Felt::from_bytes_be(&{
                    let mut data = x.0;
                    data.reverse();
                    data
                })
            })
            .collect();

            let result = ptr.deploy(
                class_hash,
                contract_address_salt,
                &calldata,
                deploy_from_zero,
                gas,
            );

            *result_ptr = match result {
                Ok(x) => {
                    let felts: Vec<_> = x.1.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
                    SyscallResultAbi {
                        ok: ManuallyDrop::new(SyscallResultAbiOk {
                            tag: 0u8,
                            payload: ManuallyDrop::new((Felt252Abi(x.0.to_bytes_le()), felts_ptr)),
                        }),
                    }
                }
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_replace_class(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            gas: &mut u128,
            class_hash: &Felt252Abi,
        ) {
            let class_hash = Felt::from_bytes_be(&{
                let mut data = class_hash.0;
                data.reverse();
                data
            });
            let result = ptr.replace_class(class_hash, gas);

            *result_ptr = match result {
                Ok(_) => SyscallResultAbi {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                        tag: 0u8,
                        payload: ManuallyDrop::new(()),
                    }),
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_library_call(
            result_ptr: &mut SyscallResultAbi<(NonNull<Felt252Abi>, u32, u32)>,
            ptr: &mut T,
            gas: &mut u128,
            class_hash: &Felt252Abi,
            function_selector: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
        ) {
            let class_hash = Felt::from_bytes_be(&{
                let mut data = class_hash.0;
                data.reverse();
                data
            });
            let function_selector = Felt::from_bytes_be(&{
                let mut data = function_selector.0;
                data.reverse();
                data
            });

            let calldata: Vec<_> = unsafe {
                let len = (*calldata).1 as usize;
                std::slice::from_raw_parts((*calldata).0, len)
            }
            .iter()
            .map(|x| {
                Felt::from_bytes_be(&{
                    let mut data = x.0;
                    data.reverse();
                    data
                })
            })
            .collect();

            let result = ptr.library_call(class_hash, function_selector, &calldata, gas);

            *result_ptr = match result {
                Ok(x) => {
                    let felts: Vec<_> = x.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
                    SyscallResultAbi {
                        ok: ManuallyDrop::new(SyscallResultAbiOk {
                            tag: 0u8,
                            payload: ManuallyDrop::new(felts_ptr),
                        }),
                    }
                }
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_call_contract(
            result_ptr: &mut SyscallResultAbi<(NonNull<Felt252Abi>, u32, u32)>,
            ptr: &mut T,
            gas: &mut u128,
            address: &Felt252Abi,
            entry_point_selector: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
        ) {
            let address = Felt::from_bytes_be(&{
                let mut data = address.0;
                data.reverse();
                data
            });
            let entry_point_selector = Felt::from_bytes_be(&{
                let mut data = entry_point_selector.0;
                data.reverse();
                data
            });

            let calldata: Vec<_> = unsafe {
                let len = (*calldata).1 as usize;
                std::slice::from_raw_parts((*calldata).0, len)
            }
            .iter()
            .map(|x| {
                Felt::from_bytes_be(&{
                    let mut data = x.0;
                    data.reverse();
                    data
                })
            })
            .collect();

            let result = ptr.call_contract(address, entry_point_selector, &calldata, gas);

            *result_ptr = match result {
                Ok(x) => {
                    let felts: Vec<_> = x.iter().map(|x| Felt252Abi(x.to_bytes_le())).collect();
                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
                    SyscallResultAbi {
                        ok: ManuallyDrop::new(SyscallResultAbiOk {
                            tag: 0u8,
                            payload: ManuallyDrop::new(felts_ptr),
                        }),
                    }
                }
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_storage_read(
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            ptr: &mut T,
            gas: &mut u128,
            address_domain: u32,
            address: &Felt252Abi,
        ) {
            let address = Felt::from_bytes_be(&{
                let mut data = address.0;
                data.reverse();
                data
            });
            let result = ptr.storage_read(address_domain, address, gas);

            *result_ptr = match result {
                Ok(res) => SyscallResultAbi {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                        tag: 0u8,
                        payload: ManuallyDrop::new(Felt252Abi(res.to_bytes_le())),
                    }),
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_storage_write(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            gas: &mut u128,
            address_domain: u32,
            address: &Felt252Abi,
            value: &Felt252Abi,
        ) {
            let address = Felt::from_bytes_be(&{
                let mut data = address.0;
                data.reverse();
                data
            });
            let value = Felt::from_bytes_be(&{
                let mut data = value.0;
                data.reverse();
                data
            });
            let result = ptr.storage_write(address_domain, address, value, gas);

            *result_ptr = match result {
                Ok(_) => SyscallResultAbi {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                        tag: 0u8,
                        payload: ManuallyDrop::new(()),
                    }),
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_emit_event(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            gas: &mut u128,
            keys: *const (*const Felt252Abi, u32, u32),
            data: *const (*const Felt252Abi, u32, u32),
        ) {
            let keys: Vec<_> = unsafe {
                let len = (*keys).1 as usize;
                std::slice::from_raw_parts((*keys).0, len)
            }
            .iter()
            .map(|x| {
                Felt::from_bytes_be(&{
                    let mut data = x.0;
                    data.reverse();
                    data
                })
            })
            .collect();

            let data: Vec<_> = unsafe {
                let len = (*data).1 as usize;
                std::slice::from_raw_parts((*data).0, len)
            }
            .iter()
            .map(|x| {
                Felt::from_bytes_be(&{
                    let mut data = x.0;
                    data.reverse();
                    data
                })
            })
            .collect();

            let result = ptr.emit_event(&keys, &data, gas);

            *result_ptr = match result {
                Ok(_) => SyscallResultAbi {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                        tag: 0u8,
                        payload: ManuallyDrop::new(()),
                    }),
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_send_message_to_l1(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            gas: &mut u128,
            to_address: &Felt252Abi,
            payload: *const (*const Felt252Abi, u32, u32),
        ) {
            let to_address = Felt::from_bytes_be(&{
                let mut data = to_address.0;
                data.reverse();
                data
            });
            let payload: Vec<_> = unsafe {
                let len = (*payload).1 as usize;
                std::slice::from_raw_parts((*payload).0, len)
            }
            .iter()
            .map(|x| {
                Felt::from_bytes_be(&{
                    let mut data = x.0;
                    data.reverse();
                    data
                })
            })
            .collect();

            let result = ptr.send_message_to_l1(to_address, &payload, gas);

            *result_ptr = match result {
                Ok(_) => SyscallResultAbi {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                        tag: 0u8,
                        payload: ManuallyDrop::new(()),
                    }),
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_keccak(
            result_ptr: &mut SyscallResultAbi<U256>,
            ptr: &mut T,
            gas: &mut u128,
            input: *const (*const u64, u32, u32),
        ) {
            let input = unsafe {
                let len = (*input).1 as usize;

                std::slice::from_raw_parts((*input).0, len)
            };

            let result = ptr.keccak(input, gas);

            *result_ptr = match result {
                Ok(x) => SyscallResultAbi {
                    ok: ManuallyDrop::new(SyscallResultAbiOk {
                        tag: 0u8,
                        payload: ManuallyDrop::new(U256(x.0)),
                    }),
                },
                Err(e) => Self::wrap_error(&e),
            };
        }
    }
}
