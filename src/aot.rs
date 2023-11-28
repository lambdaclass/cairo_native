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
    pub calldata: (*const Felt252Abi, u32, u32),
}

// !llvm.struct<(  array<0 x i8>, i128, ptr, struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>  )>

// struct<(i1, array<7 x i8>, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>

// struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>, array<0 x i8>)>

#[repr(C)]
struct RetEnum {
    tag: u8,
    // data exists if tag == 0
    data: RetEnumData,
}

#[repr(C)]
union RetEnumData {
    ok: (),
    err: (*const Felt252Abi, u32, u32),
}

#[repr(C)]
struct RetValue {
    range_check: (),
    gas: u128,
    syscall_handler: *const (),
    return_values: RetEnum,
}

pub fn call_contract_library(
    path: &Path,
    entry_point: &GenFunction<StatementIdx>,
    syscall_handler: &mut dyn StarkNetSyscallHandler,
) -> Result<(), Box<dyn Error>> {
    dbg!(&entry_point.signature);
    let symbol: &str = entry_point.id.debug_name.as_deref().unwrap();

    // todo: verify signature matches that of a contract, so unsafe is "safe"

    let felt = Felt252Abi([1; 32]);
    let payload = (addr_of!(felt), 1, 1);

    let calldata = Calldata { calldata: payload };

    unsafe {
        let lib = libloading::Library::new(path)?;

        // llvm.func @"_mlir_ciface_hello_starknet::hello_starknet::Echo::__wrapper__echo"
        // (%arg0: !llvm.ptr, %arg1: !llvm.array<0 x i8>, %arg2: i128, %arg3: !llvm.ptr, %arg4: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>)
        // attributes {llvm.emit_c_interface, sym_visibility = "public"} {

        let syscall_ptr = addr_of_mut!(*syscall_handler).cast();
        dbg!(syscall_ptr);

        let func: libloading::Symbol<
            unsafe extern "C" fn(
                range_check: (),
                gas_builtin: &u128,
                syscall_handler: *mut (),
                calldata: Calldata,
            ) -> *mut RetValue,
        > = lib.get(format!("_mlir_ciface_{}", symbol).as_bytes())?;

        let gas: u128 = u64::MAX.into();
        let result = func((), &gas, syscall_ptr, calldata);

        // fix tag, because in llvm we use tag as a i1, the padding bytes may have garbage

        println!("{:#010b}", (*result).return_values.tag);

        dbg!(gas);
        dbg!((*result).gas);
        dbg!(gas == (*result).gas);
        dbg!(gas.saturating_sub((*result).gas));
        dbg!((*result).syscall_handler);
        dbg!((*result).return_values.tag);
        dbg!((*result).return_values.tag & 0x1);

        let x = (*result).return_values.data.err;
        dbg!(x);
    }
    Ok(())
}
