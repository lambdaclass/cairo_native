use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program_registry::ProgramRegistry,
};
use cairo_native::{context::NativeContext, utils::find_function_id, values::JITValue};
use std::path::Path;

fn main() {
    let program_path = Path::new("aot_api.cairo");

    // Compile the cairo program to sierra.
    let program = cairo_native::utils::cairo_to_sierra(program_path);

    let native_context = NativeContext::new();
    let native_program = native_context.compile(&program).unwrap();

    let object_data = cairo_native::module_to_object(native_program.module()).unwrap();
    cairo_native::object_to_shared_lib(&object_data, Path::new("aot_api.dylib")).unwrap();

    let shared_lib = unsafe { libloading::Library::new("aot_api.dylib").unwrap() };

    let executor = cairo_native::aot::AotNativeExecutor::new(
        shared_lib,
        ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap(),
    );

    executor.invoke_dynamic(find_function_id(&program, "aot_api::aot_api::invoke0"), &[]);
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_felt252"),
        &[JITValue::Felt252(1234.into())],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_u8"),
        &[JITValue::Uint8(8)],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_u16"),
        &[JITValue::Uint8(16)],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_u32"),
        &[JITValue::Uint8(32)],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_u64"),
        &[JITValue::Uint8(64)],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_u128"),
        &[JITValue::Uint128(128)],
    );
    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_MyStruct"),
    //     &[JITValue::Struct {
    //         debug_name: None,
    //         fields: vec![
    //             JITValue::Felt252(b'a'.into()),
    //             JITValue::Uint8(b'b'),
    //             JITValue::Uint64(b'c' as u64),
    //         ],
    //     }],
    // );

    executor.invoke_dynamic(
        find_function_id(
            &program,
            "aot_api::aot_api::invoke8_u64_u64_u64_u64_u64_u64_u64_u64",
        ),
        &[
            JITValue::Uint8(0),
            JITValue::Uint8(1),
            JITValue::Uint8(2),
            JITValue::Uint8(3),
            JITValue::Uint8(4),
            JITValue::Uint8(5),
            JITValue::Uint8(6),
            JITValue::Uint8(7),
        ],
    );
    executor.invoke_dynamic(
        find_function_id(
            &program,
            "aot_api::aot_api::invoke9_u64_u64_u64_u64_u64_u64_u64_u64_u64",
        ),
        &[
            JITValue::Uint8(0),
            JITValue::Uint8(1),
            JITValue::Uint8(2),
            JITValue::Uint8(3),
            JITValue::Uint8(4),
            JITValue::Uint8(5),
            JITValue::Uint8(6),
            JITValue::Uint8(7),
            JITValue::Uint8(8),
        ],
    );
}
