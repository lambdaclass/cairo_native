use std::{
    alloc::Layout,
    error::Error,
    mem::ManuallyDrop,
    path::Path,
    ptr::{addr_of, addr_of_mut, NonNull},
};

use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::{GenFunction, StatementIdx},
    program_registry::ProgramRegistry,
};

use crate::{
    metadata::syscall_handler::SyscallHandlerMeta,
    starknet::{
        handler::{SyscallResultAbi, SyscallResultAbiErr, SyscallResultAbiOk},
        Felt252Abi, StarkNetSyscallHandler,
    },
    utils::{felt252_bigint, get_integer_layout},
    values::JITValue,
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

pub fn call_contract_library<T: StarkNetSyscallHandler>(
    path: &Path,
    entry_point: &GenFunction<StatementIdx>,
    syscall_handler: &mut T,
    reg: &ProgramRegistry<CoreType, CoreLibfunc>,
) -> Result<(), Box<dyn Error>> {
    // dbg!(&entry_point.signature);
    let symbol: &str = entry_point.id.debug_name.as_deref().unwrap();

    // todo: verify signature matches that of a contract, so unsafe is "safe"

    //let felt = Felt252Abi([1; 32]);
    //let payload = (addr_of!(felt), 1, 1);

    //let calldata = Calldata { calldata: payload };

    unsafe {
        let lib = libloading::Library::new(path)?;

        // llvm.func @"_mlir_ciface_hello_starknet::hello_starknet::Echo::__wrapper__echo"
        // (%arg0: !llvm.ptr, %arg1: !llvm.array<0 x i8>, %arg2: i128, %arg3: !llvm.ptr, %arg4: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>)
        // attributes {llvm.emit_c_interface, sym_visibility = "public"} {

        let arena = Bump::new();

        let ty = &entry_point.params[3].ty;
        // dbg!(ty);

        let calldata = JITValue::Struct {
            fields: vec![JITValue::Array(vec![
                JITValue::Felt252(1.into()),
                JITValue::Felt252(1.into()),
                JITValue::Felt252(1.into()),
            ])],
            debug_name: None,
        }
        .to_jit(&arena, reg, ty)
        .unwrap();

        dbg!(&calldata);

        let syscall_handler_meta = SyscallHandlerMeta::new(syscall_handler);

        let syscall_addr = syscall_handler_meta.as_ptr().as_ptr() as *const () as usize;

        let syscall_alloc = arena.alloc(syscall_addr as *mut ());

        let gas_ptr: *mut u128 = arena.alloc_layout(Layout::new::<u128>()).as_ptr().cast();
        gas_ptr.write(u64::MAX.into());

        let func: libloading::Symbol<
            unsafe extern "C" fn(
                range_check: *const (),
                gas_builtin: *mut u128,
                syscall_handler: *mut (),
                calldata: *mut (),
            ) -> *mut RetValue,
        > = lib.get(format!("_mlir_ciface_{}", symbol).as_bytes())?;

        let gas: u128 = u64::MAX.into();
        let range_check = arena.alloc_layout(Layout::new::<()>()).as_ptr().cast();
        let result = func(
            range_check,
            gas_ptr,
            syscall_alloc.cast(),
            calldata.as_ptr().cast(),
        );

        // fix tag, because in llvm we use tag as a i1, the padding bytes may have garbage

        println!("{:#010b}", (*result).return_values.tag);

        dbg!(*gas_ptr);
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
