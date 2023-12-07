use cairo_felt::Felt252;
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
        &[JITValue::Uint16(16)],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_u32"),
        &[JITValue::Uint32(32)],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_u64"),
        &[JITValue::Uint64(64)],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_u128"),
        &[JITValue::Uint128(128)],
    );

    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_MyStruct"),
        &[JITValue::Struct {
            debug_name: None,
            fields: vec![
                JITValue::Felt252(b'a'.into()),
                JITValue::Uint8(b'b'),
                JITValue::Uint64(b'c' as u64),
            ],
        }],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_Array_felt252"),
        &[JITValue::Array(vec![
            JITValue::Felt252(Felt252::from(1234)),
            JITValue::Felt252(Felt252::from(2345)),
            JITValue::Felt252(Felt252::from(3456)),
            JITValue::Felt252(Felt252::from(4567)),
        ])],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_MyEnum"),
        &[JITValue::Enum {
            tag: 0,
            value: JITValue::Uint64(0xDEADBEEFDEADBEEF).into(),
            debug_name: None,
        }],
    );
    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::invoke1_MyEnum"),
        &[JITValue::Enum {
            tag: 1,
            value: JITValue::Uint8(0xFF).into(),
            debug_name: None,
        }],
    );

    executor.invoke_dynamic(
        find_function_id(
            &program,
            "aot_api::aot_api::invoke8_u64_u64_u64_u64_u64_u64_u64_u64",
        ),
        &[
            JITValue::Uint64(0),
            JITValue::Uint64(1),
            JITValue::Uint64(2),
            JITValue::Uint64(3),
            JITValue::Uint64(4),
            JITValue::Uint64(5),
            JITValue::Uint64(6),
            JITValue::Uint64(7),
        ],
    );
    executor.invoke_dynamic(
        find_function_id(
            &program,
            "aot_api::aot_api::invoke9_u64_u64_u64_u64_u64_u64_u64_u64_u64",
        ),
        &[
            JITValue::Uint64(0),
            JITValue::Uint64(1),
            JITValue::Uint64(2),
            JITValue::Uint64(3),
            JITValue::Uint64(4),
            JITValue::Uint64(5),
            JITValue::Uint64(6),
            JITValue::Uint64(7),
            JITValue::Uint64(8),
        ],
    );
}
