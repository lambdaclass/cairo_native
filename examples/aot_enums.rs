use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
};
use cairo_native::{context::NativeContext, utils::find_function_id};
use libloading::Library;
use std::{error::Error, path::Path};

#[derive(Debug)]
enum MyEnum {
    V64(u64),
    V32(u32),
    V16(u16),
    V08(u8),
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let program_path = Path::new("aot_enums.cairo");

    // Compile the cairo program to sierra.
    let program = cairo_native::utils::cairo_to_sierra(program_path);

    let native_context = NativeContext::new();
    let native_program = native_context.compile(&program)?;

    let object_data = cairo_native::module_to_object(native_program.module())?;
    cairo_native::object_to_shared_lib(&object_data, Path::new("aot_enums.so"))?;

    let shared_lib = unsafe { libloading::Library::new("/home/dev/cairo_native/aot_enums.so")? };

    unsafe {
        call_invoke1_MyEnum(
            &program,
            native_program.program_registry(),
            &shared_lib,
            MyEnum::V64(64),
        );
        call_invoke1_MyEnum(
            &program,
            native_program.program_registry(),
            &shared_lib,
            MyEnum::V32(32),
        );
        call_invoke1_MyEnum(
            &program,
            native_program.program_registry(),
            &shared_lib,
            MyEnum::V16(16),
        );
        call_invoke1_MyEnum(
            &program,
            native_program.program_registry(),
            &shared_lib,
            MyEnum::V08(8),
        );
    }

    Ok(())
}

#[allow(non_snake_case)]
unsafe fn call_invoke1_MyEnum(
    program: &Program,
    program_registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    shared_lib: &Library,
    arg0: MyEnum,
) {
    print_signature(program, program_registry, "aot_enums::aot_enums::main");

    // Enum ABI:
    //
    //   MyEnum::V64(u64) => (u8, [u8; 7], u64, [u8;  0])
    //   MyEnum::V32(u32) => (u8, [u8; 3], u32, [u8;  8])
    //   MyEnum::V16(u16) => (u8, [u8; 1], u16, [u8; 12])
    //   MyEnum::V08(u8 ) => (u8, [u8; 0], u8,  [u8; 14])

    // match arg0 {
    //     MyEnum::V64(arg0) => {
    //         let invoke_fn = shared_lib
    //             .get::<extern "C" fn(u8, u8, u8, u8, u8, u8, u8, u8, u64)>(
    //                 b"aot_enums::aot_enums::main",
    //             )
    //             .unwrap();
    //         (invoke_fn)(0, 0, 0, 0, 0, 0, 0, 0, arg0);
    //     }
    //     MyEnum::V32(arg0) => {
    //         let invoke_fn = shared_lib
    //             .get::<extern "C" fn(u8, u8, u8, u8, u32)>(b"aot_enums::aot_enums::main")
    //             .unwrap();
    //         (invoke_fn)(1, 0, 0, 0, arg0);
    //     }
    //     MyEnum::V16(arg0) => {
    //         let invoke_fn = shared_lib
    //             .get::<extern "C" fn(u8, u8, u16)>(b"aot_enums::aot_enums::main")
    //             .unwrap();
    //         (invoke_fn)(2, 0, arg0);
    //     }
    //     MyEnum::V08(arg0) => {
    //         let invoke_fn = shared_lib
    //             .get::<extern "C" fn(u8, u8)>(b"aot_enums::aot_enums::main")
    //             .unwrap();
    //         (invoke_fn)(3, arg0);
    //     }
    // }

    match arg0 {
        MyEnum::V64(arg0) => {
            let invoke_fn = shared_lib
                .get::<extern "C" fn(u8, u8, u8, u8, u8, u8, u8, u8, u64)>(
                    b"aot_enums::aot_enums::main",
                )
                .unwrap();
            (invoke_fn)(0, 0, 0, 0, 0, 0, 0, 0, arg0);
        }
        MyEnum::V32(arg0) => {
            let invoke_fn = shared_lib
                .get::<extern "C" fn(u8, u8, u8, u8, u8, u8, u8, u8)>(b"aot_enums::aot_enums::main")
                .unwrap();

            let [l0, l1, l2, l3] = arg0.to_le_bytes();
            (invoke_fn)(1, 0, 0, 0, l0, l1, l2, l3);
        }
        MyEnum::V16(arg0) => {
            let invoke_fn = shared_lib
                .get::<extern "C" fn(u8, u8, u8, u8)>(b"aot_enums::aot_enums::main")
                .unwrap();

            let [l0, l1] = arg0.to_le_bytes();
            (invoke_fn)(2, 0, l0, l1);
        }
        MyEnum::V08(arg0) => {
            let invoke_fn = shared_lib
                .get::<extern "C" fn(u8, u8)>(b"aot_enums::aot_enums::main")
                .unwrap();
            (invoke_fn)(3, arg0);
        }
    }
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
