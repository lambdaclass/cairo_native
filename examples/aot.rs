use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program_registry::ProgramRegistry,
};
use cairo_native::{
    context::NativeContext, executor::AotNativeExecutor, utils::find_function_id, values::JitValue,
};
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

    let executor = AotNativeExecutor::new(
        shared_lib,
        ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap(),
    );

    executor.invoke_dynamic(
        find_function_id(&program, "aot_api::aot_api::contract_call"),
        &[JitValue::Array(vec![
            // Array length
            JitValue::Felt252(1.into()),
            // Call::to
            JitValue::Felt252(12345678.into()),
            // Call::selector
            JitValue::Felt252(12345678.into()),
            // Call::calldata
            JitValue::Felt252(0.into()),
        ])],
        None,
    );

    // executor.invoke_dynamic(find_function_id(&program, "aot_api::aot_api::invoke0"), &[]);
    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_felt252"),
    //     &[JitValue::Felt252(1234.into())],
    // );
    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_u8"),
    //     &[JitValue::Uint8(8)],
    // );
    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_u16"),
    //     &[JitValue::Uint16(16)],
    // );
    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_u32"),
    //     &[JitValue::Uint32(32)],
    // );
    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_u64"),
    //     &[JitValue::Uint64(64)],
    // );
    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_u128"),
    //     &[JitValue::Uint128(128)],
    // );

    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_MyStruct"),
    //     &[JitValue::Struct {
    //         debug_name: None,
    //         fields: vec![
    //             JitValue::Felt252(b'a'.into()),
    //             JitValue::Uint8(b'b'),
    //             JitValue::Uint64(b'c' as u64),
    //         ],
    //     }],
    // );
    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_Array_felt252"),
    //     &[JitValue::Array(vec![
    //         JitValue::Felt252(Felt252::from(1234)),
    //         JitValue::Felt252(Felt252::from(2345)),
    //         JitValue::Felt252(Felt252::from(3456)),
    //         JitValue::Felt252(Felt252::from(4567)),
    //     ])],
    // );
    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_MyEnum"),
    //     &[JitValue::Enum {
    //         tag: 0,
    //         value: JitValue::Uint64(0xDEADBEEFDEADBEEF).into(),
    //         debug_name: None,
    //     }],
    // );
    // executor.invoke_dynamic(
    //     find_function_id(&program, "aot_api::aot_api::invoke1_MyEnum"),
    //     &[JitValue::Enum {
    //         tag: 1,
    //         value: JitValue::Uint8(0xFF).into(),
    //         debug_name: None,
    //     }],
    // );

    // executor.invoke_dynamic(
    //     find_function_id(
    //         &program,
    //         "aot_api::aot_api::invoke8_u64_u64_u64_u64_u64_u64_u64_u64",
    //     ),
    //     &[
    //         JitValue::Uint64(0),
    //         JitValue::Uint64(1),
    //         JitValue::Uint64(2),
    //         JitValue::Uint64(3),
    //         JitValue::Uint64(4),
    //         JitValue::Uint64(5),
    //         JitValue::Uint64(6),
    //         JitValue::Uint64(7),
    //     ],
    // );
    // executor.invoke_dynamic(
    //     find_function_id(
    //         &program,
    //         "aot_api::aot_api::invoke9_u64_u64_u64_u64_u64_u64_u64_u64_u64",
    //     ),
    //     &[
    //         JitValue::Uint64(0),
    //         JitValue::Uint64(1),
    //         JitValue::Uint64(2),
    //         JitValue::Uint64(3),
    //         JitValue::Uint64(4),
    //         JitValue::Uint64(5),
    //         JitValue::Uint64(6),
    //         JitValue::Uint64(7),
    //         JitValue::Uint64(8),
    //     ],
    // );
}
