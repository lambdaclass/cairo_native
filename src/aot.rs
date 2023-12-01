use core::fmt;
use std::{
    alloc::Layout,
    error::Error,
    fmt::Debug,
    mem::{size_of, ManuallyDrop},
    path::Path,
    ptr::{addr_of, addr_of_mut, null_mut, NonNull},
};

use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::{GenFunction, StatementIdx},
    program_registry::ProgramRegistry,
};

use crate::{
    metadata::syscall_handler::SyscallHandlerMeta,
    starknet::StarkNetSyscallHandler,
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

/// Binary representation of a `felt252` (in MLIR).
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct Felt252Abi(pub [u8; 32]);

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
#[derive(Debug)]
struct RetValue {
    range_check: (),
    gas: u128,
    syscall_handler: *const (),
    return_values: RetEnum,
}

impl fmt::Debug for RetEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let res = if self.tag == 0 {
            Result::Ok(())
        } else {
            Result::Err(unsafe { self.data.err })
        };

        f.debug_struct("RetEnum")
            .field("tag", &self.tag)
            .field("data", &res)
            .finish()
    }
}

pub fn call_contract_library<T: StarkNetSyscallHandler + Debug>(
    path: &Path,
    entry_point: &GenFunction<StatementIdx>,
    syscall_handler: &mut T,
    reg: &ProgramRegistry<CoreType, CoreLibfunc>,
) -> Result<(), Box<dyn Error>> {
    // dbg!(&entry_point.signature);
    let symbol: &str = entry_point.id.debug_name.as_deref().unwrap();

    // todo: verify signature matches that of a contract, so unsafe is "safe"

    let felt: *mut Felt252Abi = unsafe {
        libc::realloc(
            null_mut(),
            Layout::array::<Felt252Abi>(1)
                .unwrap()
                .pad_to_align()
                .size(),
        )
        .cast()
    };

    unsafe {
        felt.write(Felt252Abi([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 6,
        ]));
    }
    /*
       let felt = [
           Felt252Abi([
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 6,
           ]),
           Felt252Abi([
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 6,
           ]),
       ]
       .as_slice();
    */
    let payload = (felt.cast_const(), 1u32, 1u32);

    let calldata = Calldata { calldata: payload };

    unsafe {
        let lib = libloading::Library::new(path)?;

        // llvm.func @"_mlir_ciface_hello_starknet::hello_starknet::Echo::__wrapper__echo"
        // (%arg0: !llvm.ptr, %arg1: !llvm.array<0 x i8>, %arg2: i128, %arg3: !llvm.ptr, %arg4: !llvm.struct<(struct<(ptr<i252>, i32, i32)>)>)
        // attributes {llvm.emit_c_interface, sym_visibility = "public"} {

        let arena = Bump::new();

        //let ty = &entry_point.params[3].ty;
        // dbg!(ty);

        /*
        let calldata2 = JITValue::Struct {
            fields: vec![JITValue::Array(vec![JITValue::Felt252(1.into())])],
            debug_name: None,
        }
        .to_jit(&arena, reg, ty)
        .unwrap();

        */

        dbg!(&calldata);

        let syscall_handler_meta = SyscallHandlerMeta::new(syscall_handler);
        let syscall_addr = syscall_handler_meta.as_ptr().as_ptr();

        let return_value = arena.alloc_layout(Layout::new::<RetValue>()).cast();

        let func: libloading::Symbol<
            unsafe extern "C" fn(
                return_value: *mut RetValue,
                range_check: [u8; 0],
                gas_builtin: u128,
                syscall_handler: *mut std::ffi::c_void,
                calldata: Calldata,
            ),
        > = lib.get(format!("_mlir_ciface_{}\0", symbol).as_bytes())?;

        let gas: u128 = 10000000000;
        func(
            return_value.as_ptr(),
            [],
            gas,
            syscall_addr.cast(),
            calldata,
        );

        let return_value = return_value.as_ptr();

        println!("{:#010b}", (*return_value).return_values.tag);

        let res_data = return_value.as_ref().unwrap();
        dbg!(return_value);
        dbg!(res_data);
        dbg!(gas.saturating_sub((*return_value).gas));

        std::mem::forget(arena);
    }
    Ok(())
}
