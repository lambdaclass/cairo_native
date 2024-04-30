#![no_main]
use starknet_types_core::felt::Felt;
use cairo_native::context::NativeContext;
use cairo_native::executor::NativeExecutor;
use cairo_native::values::JitValue;
use std::path::Path;


// create a bunch of useful functions to make it easy to fuzz .sierra programs
// create a basic .sierra program to test the fuzzer with
//

fuzz_target!(|data: &[u8]| {
    // fuzzed code goes here
});

fn build_program(program_path: &str) {
    let path = Path::new(program_path);

    let contract = compile_path(
        path,
        None,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let entry_point = contract.entry_points_by_type.constructor.get(0).unwrap();
    let sierra_program = contract.extract_sierra_program().unwrap();

    let native_context = NativeContext::new();

    let mut native_program = native_context.compile(&sierra_program).unwrap();
    native_program
        .insert_metadata(SyscallHandlerMeta::new(&mut SyscallHandler))
        .unwrap();

    // Call the echo function from the contract using the generated wrapper.
    let entry_point_fn =
        find_entry_point_by_idx(&sierra_program, entry_point.function_idx).unwrap();

    let fn_id = &entry_point_fn.id;

    let native_executor = NativeExecutor::new(native_program);

    let params =  &[JitValue::Felt252(Felt::from(1))];

    let result = native_executor
        .execute_contract(
            fn_id,
           params,
            u64::MAX.into(),
        )
        .expect("failed to execute the given contract");

    println!();
    println!("Cairo program was compiled and executed successfully.");
    println!("{result:#?}");
}
