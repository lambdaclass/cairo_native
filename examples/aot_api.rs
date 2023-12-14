#![cfg(target_arch = "aarch64")]

use cairo_felt::Felt252;
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
};
use cairo_native::{context::NativeContext, starknet::Felt252Abi, utils::find_function_id};
use libloading::Library;
use std::{arch::asm, error::Error, path::Path};

#[derive(Debug)]
#[repr(C, align(16))]
struct MyStruct {
    a: [u64; 4],
    b: u8,
    c: u64,
}

#[derive(Debug)]
enum MyEnum {
    A(u64),
    B(u8),
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let program_path = Path::new("aot_api.cairo");

    // Compile the cairo program to sierra.
    let program = cairo_native::utils::cairo_to_sierra(program_path);

    let native_context = NativeContext::new();
    let native_program = native_context.compile(&program)?;

    let object_data = cairo_native::module_to_object(native_program.module())?;
    cairo_native::object_to_shared_lib(&object_data, Path::new("aot_api.dylib"))?;

    let shared_lib = unsafe { libloading::Library::new("aot_api.dylib")? };

    unsafe {
        call_invoke0(&program, native_program.program_registry(), &shared_lib);
    }

    unsafe {
        call_invoke1_felt252(
            &program,
            native_program.program_registry(),
            &shared_lib,
            42.into(),
        );
    }

    unsafe {
        call_invoke1_u8(&program, native_program.program_registry(), &shared_lib, 42);
    }

    unsafe {
        call_invoke1_u16(&program, native_program.program_registry(), &shared_lib, 42);
    }

    unsafe {
        call_invoke1_u32(&program, native_program.program_registry(), &shared_lib, 42);
    }

    unsafe {
        call_invoke1_u64(&program, native_program.program_registry(), &shared_lib, 42);
    }

    unsafe {
        call_invoke1_u64(&program, native_program.program_registry(), &shared_lib, 42);
    }

    unsafe {
        call_invoke1_MyStruct(
            &program,
            native_program.program_registry(),
            &shared_lib,
            MyStruct {
                a: Felt252::from(1234).to_le_digits(),
                b: 100,
                c: 158734029865403,
            },
        );
    }

    unsafe {
        call_invoke1_Array_felt252(
            &program,
            native_program.program_registry(),
            &shared_lib,
            &[
                Felt252Abi(Felt252::from(1).to_le_bytes()),
                Felt252Abi(Felt252::from(2).to_le_bytes()),
                Felt252Abi(Felt252::from(3).to_le_bytes()),
            ],
        );
    }

    unsafe {
        call_invoke1_MyEnum(
            &program,
            native_program.program_registry(),
            &shared_lib,
            MyEnum::A(0xDEADBEEF),
        );
        call_invoke1_MyEnum(
            &program,
            native_program.program_registry(),
            &shared_lib,
            MyEnum::B(0x10),
        );
    }

    unsafe {
        call_invoke2_MyEnum_MyStruct(
            &program,
            native_program.program_registry(),
            &shared_lib,
            MyEnum::A(0xDEADBEEF),
            MyStruct {
                a: Felt252::from(1234).to_le_digits(),
                b: 100,
                c: 158734029865403,
            },
        );
    }

    unsafe {
        call_invoke5_u64_felt252_felt252_felt252_felt252(
            &program,
            native_program.program_registry(),
            &shared_lib,
            42,
            Felt252::from(1),
            Felt252::from(2),
            Felt252::from(3),
            Felt252::from(4),
        );
    }

    unsafe {
        call_invoke0_return1_felt252(&program, native_program.program_registry(), &shared_lib);
    }

    unsafe {
        call_invoke0_return1_u64(&program, native_program.program_registry(), &shared_lib);
    }

    unsafe {
        call_invoke0_return1_tuple8_u64(&program, native_program.program_registry(), &shared_lib);
    }

    unsafe {
        call_invoke0_return1_tuple10_u64(&program, native_program.program_registry(), &shared_lib);
    }

    Ok(())
}

unsafe fn call_invoke0(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
) {
    print_signature(program, program_registry, "aot_api::aot_api::invoke0");

    let invoked_fn = shared_lib
        .get::<extern "C" fn()>(b"_mlir_ciface_aot_api::aot_api::invoke0")
        .unwrap();

    (invoked_fn)();
}

unsafe fn call_invoke1_felt252(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: Felt252,
) {
    print_signature(
        program,
        program_registry,
        "aot_api::aot_api::invoke1_felt252",
    );

    let arg0 = arg0.to_le_digits();

    let invoked_fn = shared_lib
        .get::<extern "C" fn(u8, u64, u64, u64, u64)>(
            b"_mlir_ciface_aot_api::aot_api::invoke1_felt252",
        )
        .unwrap();

    invoked_fn(0, arg0[0], arg0[1], arg0[2], arg0[3]);
}

unsafe fn call_invoke1_u8(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: u8,
) {
    print_signature(program, program_registry, "aot_api::aot_api::invoke1_u8");

    let invoked_fn = shared_lib
        .get::<extern "C" fn(u8, u8)>(b"_mlir_ciface_aot_api::aot_api::invoke1_u8")
        .unwrap();

    invoked_fn(0, arg0);
}

unsafe fn call_invoke1_u16(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: u16,
) {
    print_signature(program, program_registry, "aot_api::aot_api::invoke1_u16");

    let invoked_fn = shared_lib
        .get::<extern "C" fn(u8, u16)>(b"_mlir_ciface_aot_api::aot_api::invoke1_u16")
        .unwrap();

    invoked_fn(0, arg0);
}

unsafe fn call_invoke1_u32(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: u32,
) {
    print_signature(program, program_registry, "aot_api::aot_api::invoke1_u32");

    let invoked_fn = shared_lib
        .get::<extern "C" fn(u8, u32)>(b"_mlir_ciface_aot_api::aot_api::invoke1_u32")
        .unwrap();

    invoked_fn(0, arg0);
}

unsafe fn call_invoke1_u64(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: u64,
) {
    print_signature(program, program_registry, "aot_api::aot_api::invoke1_u64");

    let invoked_fn = shared_lib
        .get::<extern "C" fn(u8, u64)>(b"_mlir_ciface_aot_api::aot_api::invoke1_u64")
        .unwrap();

    invoked_fn(0, arg0);
}

#[allow(non_snake_case)]
unsafe fn call_invoke1_MyStruct(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: MyStruct,
) {
    print_signature(
        program,
        program_registry,
        "aot_api::aot_api::invoke1_MyStruct",
    );

    let invoked_fn = shared_lib
        .get::<extern "C" fn(u8, u64, u64, u64, u64, u8, u64)>(
            b"_mlir_ciface_aot_api::aot_api::invoke1_MyStruct",
        )
        .unwrap();

    invoked_fn(
        0, arg0.a[0], arg0.a[1], arg0.a[2], arg0.a[3], arg0.b, arg0.c,
    );
}

#[allow(non_snake_case)]
unsafe fn call_invoke1_Array_felt252(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: &[Felt252Abi],
) {
    print_signature(
        program,
        program_registry,
        "aot_api::aot_api::invoke1_Array_felt252",
    );

    let invoked_fn = shared_lib
        .get::<extern "C" fn(u8, *const Felt252Abi, u32, u32)>(
            b"_mlir_ciface_aot_api::aot_api::invoke1_Array_felt252",
        )
        .unwrap();

    invoked_fn(0, arg0.as_ptr(), arg0.len() as u32, arg0.len() as u32);
}

#[allow(non_snake_case)]
unsafe fn call_invoke1_MyEnum(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: MyEnum,
) {
    print_signature(
        program,
        program_registry,
        "aot_api::aot_api::invoke1_MyEnum",
    );

    #[repr(C, align(8))]
    struct MyEnumAbi {
        tag: u8,
        value: MyEnumValueAbi,
    }
    #[repr(C, align(8))]
    union MyEnumValueAbi {
        a: u64,
        b: u8,
    }

    let arg0 = match arg0 {
        MyEnum::A(value) => MyEnumAbi {
            tag: 0,
            value: MyEnumValueAbi { a: value },
        },
        MyEnum::B(value) => MyEnumAbi {
            tag: 1,
            value: MyEnumValueAbi { b: value },
        },
    };

    let invoked_fn = shared_lib
        .get::<extern "C" fn(u8, u8, u64, u64, u64, u64, u64, u64, u64, u64)>(
            b"_mlir_ciface_aot_api::aot_api::invoke1_MyEnum",
        )
        .unwrap();

    // Note: the tag is a normal argument, but the payload is always stored in the stack, therefore
    //   needing 7 padding arguments in this case.
    invoked_fn(
        0,
        arg0.tag, // <-- Tag.
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        arg0.value.a, // <-- Payload.
    );
}

#[allow(non_snake_case)]
unsafe fn call_invoke2_MyEnum_MyStruct(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: MyEnum,
    arg1: MyStruct,
) {
    print_signature(
        program,
        program_registry,
        "aot_api::aot_api::invoke2_MyEnum_MyStruct",
    );

    #[repr(C, align(8))]
    struct MyEnumAbi {
        tag: u8,
        value: MyEnumValueAbi,
    }
    #[repr(C, align(8))]
    union MyEnumValueAbi {
        a: u64,
        b: u8,
    }

    let arg0 = match arg0 {
        MyEnum::A(value) => MyEnumAbi {
            tag: 0,
            value: MyEnumValueAbi { a: value },
        },
        MyEnum::B(value) => MyEnumAbi {
            tag: 0,
            value: MyEnumValueAbi { b: value },
        },
    };

    let invoked_fn =
        shared_lib
            .get::<extern "C" fn(
                u8,
                u8,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u64,
                u8,
                u64,
            )>(b"_mlir_ciface_aot_api::aot_api::invoke2_MyEnum_MyStruct")
            .unwrap();

    // Note: the tag is a normal argument, but the payload is always stored in the stack, therefore
    //   needing 7 padding arguments in this case.
    invoked_fn(
        0,
        arg0.tag, // <-- Tag.
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        arg0.value.a, // <-- Payload.
        arg1.a[0],
        arg1.a[1],
        arg1.a[2],
        arg1.a[3],
        arg1.b,
        arg1.c,
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn call_invoke5_u64_felt252_felt252_felt252_felt252(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: u64,
    arg1: Felt252,
    arg2: Felt252,
    arg3: Felt252,
    arg4: Felt252,
) {
    print_signature(
        program,
        program_registry,
        "aot_api::aot_api::invoke5_u64_felt252_felt252_felt252_felt252",
    );

    let arg1 = arg1.to_le_digits();
    let arg2 = arg2.to_le_digits();
    let arg3 = arg3.to_le_digits();
    let arg4 = arg4.to_le_digits();

    let invoked_fn = shared_lib
        .get::<extern "C" fn(
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
        )>(b"_mlir_ciface_aot_api::aot_api::invoke5_u64_felt252_felt252_felt252_felt252")
        .unwrap();

    invoked_fn(
        0, arg0, arg1[0], arg1[1], arg1[2], arg1[3], arg2[0], arg2[1], arg2[2], arg2[3], arg3[0],
        arg3[1], arg3[2], arg3[3], arg4[0], arg4[1], arg4[2], arg4[3],
    );
}

unsafe fn call_invoke0_return1_felt252(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
) {
    print_signature(
        program,
        program_registry,
        "aot_api::aot_api::invoke0_return1_felt252",
    );

    let invoked_fn = shared_lib
        .get::<extern "C" fn(u8, *mut Felt252Abi)>(
            b"_mlir_ciface_aot_api::aot_api::invoke0_return1_felt252",
        )
        .unwrap();

    // (invoked_fn)(0, ret_ptr.as_mut_ptr());
    let mut l0: u64;
    let mut l1: u64;
    let mut l2: u64;
    let mut l3: u64;
    asm!(
        "blr {invoked_fn}",
        invoked_fn = in(reg) invoked_fn.into_raw().into_raw(),
        out("x0") l0,
        out("x1") l1,
        out("x2") l2,
        out("x3") l3,
    );

    let bytes: [u8; 32] = std::mem::transmute([l0, l1, l2, l3]);
    dbg!(Felt252::from_bytes_le(&bytes));
}

unsafe fn call_invoke0_return1_u64(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
) {
    print_signature(
        program,
        program_registry,
        "aot_api::aot_api::invoke0_return1_u64",
    );

    let invoked_fn = shared_lib
        .get::<extern "C" fn() -> u64>(b"_mlir_ciface_aot_api::aot_api::invoke0_return1_u64")
        .unwrap();

    dbg!((invoked_fn)());
}

unsafe fn call_invoke0_return1_tuple8_u64(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
) {
    print_signature(
        program,
        program_registry,
        "aot_api::aot_api::invoke0_return1_tuple8_u64",
    );

    let invoked_fn = shared_lib
        .get::<extern "C" fn()>(b"_mlir_ciface_aot_api::aot_api::invoke0_return1_tuple8_u64")
        .unwrap();

    let mut l = [0u64; 8];
    asm!(
        "blr {invoked_fn}",
        invoked_fn = in(reg) invoked_fn.into_raw().into_raw(),
        in("x0") std::ptr::addr_of_mut!(l),
    );

    dbg!(l);
}

unsafe fn call_invoke0_return1_tuple10_u64(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
) {
    print_signature(
        program,
        program_registry,
        "aot_api::aot_api::invoke0_return1_tuple10_u64",
    );

    let invoked_fn = shared_lib
        .get::<extern "C" fn() -> u64>(
            b"_mlir_ciface_aot_api::aot_api::invoke0_return1_tuple10_u64",
        )
        .unwrap();

    let mut l = [0u64; 10];
    asm!(
        "blr {invoked_fn}",
        in("x0") std::ptr::addr_of_mut!(l),
        invoked_fn = in(reg) invoked_fn.into_raw().into_raw(),
    );

    dbg!(l);
}

fn print_signature(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function_name: &str,
) {
    let invoke_fn_id = find_function_id(program, function_name);
    let invoke_fn_abi = program_registry.get_function(invoke_fn_id).unwrap();

    println!("Target function: {function_name}");
    println!("  Arguments:");
    for (idx, arg) in invoke_fn_abi.signature.param_types.iter().enumerate() {
        println!("    #{idx}: {}", arg.debug_name.as_deref().unwrap());
    }
    println!("  Return values:");
    for (idx, arg) in invoke_fn_abi.signature.ret_types.iter().enumerate() {
        println!("    #{idx}: {}", arg.debug_name.as_deref().unwrap());
    }
    println!();
}
