use cairo_lang_runner::StarknetState;
use cairo_native::{
    context::NativeContext,
    executor::{AotNativeExecutor, JitNativeExecutor},
    OptLevel,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use util::{create_vm_runner, prepare_programs};

mod util;

pub fn bench_libfuncs(c: &mut Criterion) {
    let programs = prepare_programs("tests/cases");

    {
        let mut c = c.benchmark_group("Libfunc Execution Time");

        for (program, filename) in &programs {
            let entry = program
                .funcs
                .iter()
                .find(|f| {
                    if let Some(name) = &f.id.debug_name {
                        name.ends_with("main")
                    } else {
                        false
                    }
                })
                .expect("failed to find entry point");

            let vm_runner = create_vm_runner(program);

            c.bench_with_input(
                BenchmarkId::new(filename, "SierraCasmRunner"),
                &program,
                |b, _program| {
                    b.iter(|| {
                        let res = vm_runner
                            .run_function_with_starknet_context(
                                entry,
                                vec![],
                                Some(usize::MAX),
                                StarknetState::default(),
                            )
                            .expect("should run correctly");
                        black_box(res)
                    })
                },
            );

            c.bench_with_input(
                BenchmarkId::new(filename, "jit-cold"),
                &program,
                |b, program| {
                    let native_context = NativeContext::new();
                    b.iter(|| {
                        let module = native_context
                            .compile(program, false, Some(Default::default()), None)
                            .unwrap();
                        // pass manager internally verifies the MLIR output is correct.
                        let native_executor =
                            JitNativeExecutor::from_native_module(module, OptLevel::Aggressive)
                                .unwrap();

                        // Execute the program.
                        let result = native_executor
                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX))
                            .unwrap();
                        black_box(result)
                    })
                },
            );

            c.bench_with_input(
                BenchmarkId::new(filename, "jit-hot"),
                program,
                |b, program| {
                    let native_context = NativeContext::new();
                    let module = native_context
                        .compile(program, false, Some(Default::default()), None)
                        .unwrap();
                    // pass manager internally verifies the MLIR output is correct.
                    let native_executor =
                        JitNativeExecutor::from_native_module(module, OptLevel::Aggressive)
                            .unwrap();

                    // warmup
                    for _ in 0..5 {
                        native_executor
                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX))
                            .unwrap();
                    }

                    b.iter(|| {
                        // Execute the program.
                        let result = native_executor
                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX))
                            .unwrap();
                        black_box(result)
                    })
                },
            );

            c.bench_with_input(
                BenchmarkId::new(filename, "aot-with-compile"),
                &program,
                |b, program| {
                    let native_context = NativeContext::new();
                    b.iter(|| {
                        let module = native_context
                            .compile(program, false, Some(Default::default()), None)
                            .unwrap();
                        // pass manager internally verifies the MLIR output is correct.
                        let native_executor =
                            AotNativeExecutor::from_native_module(module, OptLevel::Aggressive)
                                .unwrap();

                        // Execute the program.
                        let result = native_executor
                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX))
                            .unwrap();
                        black_box(result)
                    })
                },
            );

            c.bench_with_input(
                BenchmarkId::new(filename, "aot-without-compile"),
                program,
                |b, program| {
                    let native_context = NativeContext::new();
                    let module = native_context
                        .compile(program, false, Some(Default::default()), None)
                        .unwrap();
                    // pass manager internally verifies the MLIR output is correct.
                    let native_executor =
                        AotNativeExecutor::from_native_module(module, OptLevel::Aggressive)
                            .unwrap();

                    // warmup
                    for _ in 0..5 {
                        native_executor
                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX))
                            .unwrap();
                    }

                    b.iter(|| {
                        // Execute the program.
                        let result = native_executor
                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX))
                            .unwrap();
                        black_box(result)
                    })
                },
            );
        }
    }
}

criterion_group!(benches, bench_libfuncs);
criterion_main!(benches);
