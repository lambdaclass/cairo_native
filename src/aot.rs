use std::{
    error::Error,
    mem::ManuallyDrop,
    path::Path,
    ptr::{addr_of, addr_of_mut},
};

use cairo_lang_sierra::program::{GenFunction, StatementIdx};

use crate::starknet::{
    handler::{SyscallResultAbi, SyscallResultAbiErr, SyscallResultAbiOk},
    Felt252Abi, StarkNetSyscallHandler,
};

#[derive(Debug)]
#[repr(C)]
struct ResultError {
    ptr: *const Felt252Abi,
    len: u32,
    cap: u32,
}

#[derive(Debug)]
#[repr(C)]
pub struct Calldata {
    pub calldata: *const (*const Felt252Abi, u32, u32),
}

pub fn call_contract_library(
    path: &Path,
    entry_point: &GenFunction<StatementIdx>,
    syscall_handler: &mut dyn StarkNetSyscallHandler,
) -> Result<(), Box<dyn Error>> {
    dbg!(&entry_point.signature);
    let symbol: &str = entry_point.id.debug_name.as_deref().unwrap();

    // todo: verify signature matches that of a contract, so unsafe is "safe"

    let felt = Felt252Abi([0; 32]);
    let payload = (addr_of!(felt), 1, 1);

    let calldata = Calldata {
        calldata: addr_of!(payload),
    };

    unsafe {
        let lib = libloading::Library::new(path)?;
        let func: libloading::Symbol<
            unsafe extern "C" fn(
                range_check: (),
                gas_builtin: &mut u128,
                syscall_handler: *mut (),
                calldata: Calldata,
            ) -> SyscallResultAbi<()>,
        > = lib.get(format!("_mlir_ciface_{}", symbol).as_bytes())?;

        let mut gas: u128 = u64::MAX.into();
        let x = func(
            (),
            &mut gas,
            addr_of_mut!(*syscall_handler).cast(),
            calldata,
        );
        dbg!(x.ok.tag);
        dbg!(x.err.tag);
    }
    Ok(())
}
