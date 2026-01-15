use afl::fuzz;
use arbitrary::{Arbitrary, Unstructured};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program_registry::ProgramRegistry,
};
use cairo_native::{
    context::NativeContext, executor::AotNativeExecutor, include_program,
    starknet_stub::StubSyscallHandler, OptLevel,
};
use cairo_native_fuzz::{arbitrary_value, is_builtin, is_supported};

#[cfg(fuzzing)]
use afl::{ijon_inc, ijon_set};

fn main() {
    let program = include_program!("../test_data_artifacts/programs/corelib.sierra.json")
        .into_v1()
        .unwrap()
        .program;

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

        if !cfg!(fuzzing) {
            println!("function {}", func.id);
        }

        let mut param_tys = vec![];
        for param in &func.params {
            let param_ty = registry.get_type(&param.ty).unwrap();

            if !cfg!(fuzzing) {
                println!("- param {}", param.ty);
            }

            if is_builtin(param_ty) {
                continue;
            } else if is_supported(param_ty, &registry) {
                param_tys.push(param_ty)
            } else {
                return;
            };
        }

        if func
            .signature
            .ret_types
            .iter()
            .filter(|ret_ty_id| {
                let ret_ty = registry.get_type(ret_ty_id).unwrap();
                !is_builtin(ret_ty)
            })
            .count()
            > 1
        {
            return;
        }

        for ret_ty_id in &func.signature.ret_types {
            if !cfg!(fuzzing) {
                println!("- ret {}", ret_ty_id);
            }
        }

        #[cfg(fuzzing)]
        ijon_set!(func_idx as u32);

        let mut values = vec![];
        for param_ty in param_tys {
            let Ok(value) = arbitrary_value(param_ty, &mut unstructured, &registry) else {
                return;
            };

            values.push(value);

            #[cfg(fuzzing)]
            ijon_inc!(func_idx as u32);
        }

        if !cfg!(fuzzing) {
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
    });
}
