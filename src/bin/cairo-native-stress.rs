use std::fs;

use cairo_lang_sierra::program_registry::ProgramRegistry;
use cairo_lang_starknet::compile::compile_path;
use cairo_native::{
    context::NativeContext,
    executor::AotNativeExecutor,
    metadata::gas::GasMetadata,
    module_to_object, object_to_shared_lib,
    starknet::DummySyscallHandler,
    utils::{find_entry_point_by_idx, SHARED_LIBRARY_EXT},
};
use libloading::Library;

fn main() {
    let program = generate_program("Name", 252);

    let native_context = NativeContext::new();
    let native_module = native_context
        .compile(&program, None)
        .expect("should compile");

    let object_data =
        module_to_object(native_module.module(), cairo_native::OptLevel::None).unwrap();
    let shared_library_path = tempfile::Builder::new()
        .prefix("lib")
        .suffix(SHARED_LIBRARY_EXT)
        .tempfile()
        .unwrap()
        .into_temp_path();
    object_to_shared_lib(&object_data, &shared_library_path).unwrap();
    let shared_library = unsafe { Library::new(shared_library_path).unwrap() };

    let registry = ProgramRegistry::new(&program).unwrap();

    let executor = AotNativeExecutor::new(
        shared_library,
        registry,
        native_module
            .metadata()
            .get::<GasMetadata>()
            .cloned()
            .unwrap(),
    );

    let entry_point_id = &find_entry_point_by_idx(&program, 0).unwrap().id;
    let execution_result = executor
        .invoke_contract_dynamic(entry_point_id, &[], Some(u128::MAX), DummySyscallHandler)
        .unwrap();

    assert!(
        execution_result.failure_flag == false,
        "contract execution failed"
    )
}

fn generate_program(name: &str, output: u32) -> cairo_lang_sierra::program::Program {
    let program_str = format!(
        "\
#[starknet::contract]
mod {name} {{
    #[storage]
    struct Storage {{}}

    #[external(v0)]
    fn main(self: @ContractState) -> felt252 {{
        return {output};
    }}
}}
"
    );

    let mut program_file = tempfile::Builder::new()
        .prefix("test_")
        .suffix(".cairo")
        .tempfile()
        .unwrap();
    fs::write(&mut program_file, program_str).unwrap();

    let contract_class = compile_path(program_file.path(), None, Default::default()).unwrap();

    let program = contract_class.extract_sierra_program().unwrap();

    program
}
