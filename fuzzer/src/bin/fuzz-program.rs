use afl::fuzz;
use arbitrary::{Arbitrary, Unstructured};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
};
use cairo_native::{
    context::NativeContext, executor::AotNativeExecutor, starknet_stub::StubSyscallHandler,
    OptLevel,
};
use cairo_native_fuzzer::{arbitrary_value, is_function_supported};
use clap::Parser;
use std::{fs::File, path::PathBuf};

#[cfg(fuzzing)]
use afl::{ijon_inc, ijon_set};

#[derive(Parser, Debug)]
struct Args {
    sierra_path: PathBuf,
}

fn main() {
    let args = Args::parse();

    let program_file = File::open(args.sierra_path).unwrap();
    let program: Program = serde_json::from_reader(program_file).unwrap();

    let context = NativeContext::new();
    let module = context.compile(&program, false, None, None).unwrap();
    let executor = AotNativeExecutor::from_native_module(module, OptLevel::None).unwrap();

    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

    fuzz!(|data: &[u8]| {
        let mut unstructured = Unstructured::new(data);
        let Ok(func_idx) = usize::arbitrary(&mut unstructured) else {
            return;
        };
        let Some(func) = program.funcs.get(func_idx) else {
            return;
        };
        if !is_function_supported(func, &registry) {
            return;
        };

        #[cfg(fuzzing)]
        ijon_set!(func_idx as u32);

        let mut values = vec![];
        for param_ty_id in &func.signature.param_types {
            let param_ty = registry.get_type(param_ty_id).unwrap();
            let Ok(value) = arbitrary_value(param_ty, &mut unstructured, &registry) else {
                return;
            };

            values.push(value);

            #[cfg(fuzzing)]
            ijon_inc!(func_idx as u32);
        }

        if !cfg!(fuzzing) {
            println!("function {}", func.id);
            for param in &func.signature.param_types {
                println!("- param {}", param);
            }
            for ret_ty_id in &func.signature.ret_types {
                println!("- ret {}", ret_ty_id);
            }
            println!("arguments");
            for value in &values {
                println!("- {:?}", value);
            }
        }

        executor
            .invoke_dynamic_with_syscall_handler(
                &func.id,
                &values,
                Some(u64::MAX),
                &mut StubSyscallHandler::default(),
            )
            .unwrap();

        if !cfg!(fuzzing) {
            println!("ok");
        }
    });
}
