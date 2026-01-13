use cairo_native::{
    context::NativeContext, executor::AotNativeExecutor, include_program,
    starknet_stub::StubSyscallHandler, OptLevel, Value,
};
use honggfuzz::{
    arbitrary::{Arbitrary, Unstructured},
    fuzz,
};

fn main() {
    let program = include_program!("../test_data_artifacts/programs/corelib.sierra.json")
        .into_v1()
        .unwrap()
        .program;

    let context = NativeContext::new();
    let module = context.compile(&program, false, None, None).unwrap();
    let executor = AotNativeExecutor::from_native_module(module, OptLevel::Aggressive).unwrap();

    loop {
        fuzz!(|data: &[u8]| {
            let func_id = &program
                .funcs
                .iter()
                .find(|x| x.id.debug_name.as_deref() == Some("core::integer::u8_wrapping_add"))
                .expect("Test program entry point not found.")
                .id;

            let mut unstructured = Unstructured::new(data);

            let Ok(v1) = u8::arbitrary(&mut unstructured) else {
                return;
            };
            let Ok(v2) = u8::arbitrary(&mut unstructured) else {
                return;
            };

            let execution = executor
                .invoke_dynamic_with_syscall_handler(
                    func_id,
                    &[Value::Uint8(v1), Value::Uint8(v2)],
                    Some(u64::MAX),
                    &mut StubSyscallHandler::default(),
                )
                .unwrap();

            assert_eq!(execution.return_value, Value::Uint8(v1.wrapping_add(v2)))
        });
    }
}
