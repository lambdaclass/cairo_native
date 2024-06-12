//use cairo_lang_runner::StarknetState;
use cairo_lang_runner::StarknetState;
//use cairo_native::{context::NativeContext, executor::JitNativeExecutor};
use cairo_native::{context::NativeContext, executor::JitNativeExecutor};
//use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
//use util::{create_vm_runner, prepare_programs};
use util::{create_vm_runner, prepare_programs};
//

//mod util;
mod util;
//

//pub fn bench_libfuncs(c: &mut Criterion) {
pub fn bench_libfuncs(c: &mut Criterion) {
//    let programs = prepare_programs("tests/cases");
    let programs = prepare_programs("tests/cases");
//

//    {
    {
//        let mut c = c.benchmark_group("Libfunc Execution Time");
        let mut c = c.benchmark_group("Libfunc Execution Time");
//

//        for (program, filename) in &programs {
        for (program, filename) in &programs {
//            let entry = program
            let entry = program
//                .funcs
                .funcs
//                .iter()
                .iter()
//                .find(|f| {
                .find(|f| {
//                    if let Some(name) = &f.id.debug_name {
                    if let Some(name) = &f.id.debug_name {
//                        name.ends_with("main")
                        name.ends_with("main")
//                    } else {
                    } else {
//                        false
                        false
//                    }
                    }
//                })
                })
//                .expect("failed to find entry point");
                .expect("failed to find entry point");
//

//            let vm_runner = create_vm_runner(program);
            let vm_runner = create_vm_runner(program);
//

//            c.bench_with_input(
            c.bench_with_input(
//                BenchmarkId::new(filename, "SierraCasmRunner"),
                BenchmarkId::new(filename, "SierraCasmRunner"),
//                &program,
                &program,
//                |b, _program| {
                |b, _program| {
//                    b.iter(|| {
                    b.iter(|| {
//                        let res = vm_runner
                        let res = vm_runner
//                            .run_function_with_starknet_context(
                            .run_function_with_starknet_context(
//                                entry,
                                entry,
//                                &[],
                                &[],
//                                Some(usize::MAX),
                                Some(usize::MAX),
//                                StarknetState::default(),
                                StarknetState::default(),
//                            )
                            )
//                            .expect("should run correctly");
                            .expect("should run correctly");
//                        black_box(res)
                        black_box(res)
//                    })
                    })
//                },
                },
//            );
            );
//

//            c.bench_with_input(
            c.bench_with_input(
//                BenchmarkId::new(filename, "jit-cold"),
                BenchmarkId::new(filename, "jit-cold"),
//                &program,
                &program,
//                |b, program| {
                |b, program| {
//                    let native_context = NativeContext::new();
                    let native_context = NativeContext::new();
//                    b.iter(|| {
                    b.iter(|| {
//                        let module = native_context.compile(program, None).unwrap();
                        let module = native_context.compile(program, None).unwrap();
//                        // pass manager internally verifies the MLIR output is correct.
                        // pass manager internally verifies the MLIR output is correct.
//                        let native_executor =
                        let native_executor =
//                            JitNativeExecutor::from_native_module(module, Default::default());
                            JitNativeExecutor::from_native_module(module, Default::default());
//

//                        // Execute the program.
                        // Execute the program.
//                        let result = native_executor
                        let result = native_executor
//                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX as u128))
                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX as u128))
//                            .unwrap();
                            .unwrap();
//                        black_box(result)
                        black_box(result)
//                    })
                    })
//                },
                },
//            );
            );
//

//            c.bench_with_input(
            c.bench_with_input(
//                BenchmarkId::new(filename, "jit-hot"),
                BenchmarkId::new(filename, "jit-hot"),
//                program,
                program,
//                |b, program| {
                |b, program| {
//                    let native_context = NativeContext::new();
                    let native_context = NativeContext::new();
//                    let module = native_context.compile(program, None).unwrap();
                    let module = native_context.compile(program, None).unwrap();
//                    // pass manager internally verifies the MLIR output is correct.
                    // pass manager internally verifies the MLIR output is correct.
//                    let native_executor =
                    let native_executor =
//                        JitNativeExecutor::from_native_module(module, Default::default());
                        JitNativeExecutor::from_native_module(module, Default::default());
//

//                    // warmup
                    // warmup
//                    for _ in 0..5 {
                    for _ in 0..5 {
//                        native_executor
                        native_executor
//                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX as u128))
                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX as u128))
//                            .unwrap();
                            .unwrap();
//                    }
                    }
//

//                    b.iter(|| {
                    b.iter(|| {
//                        // Execute the program.
                        // Execute the program.
//                        let result = native_executor
                        let result = native_executor
//                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX as u128))
                            .invoke_dynamic(&entry.id, &[], Some(u64::MAX as u128))
//                            .unwrap();
                            .unwrap();
//                        black_box(result)
                        black_box(result)
//                    })
                    })
//                },
                },
//            );
            );
//        }
        }
//    }
    }
//}
}
//

//criterion_group!(benches, bench_libfuncs);
criterion_group!(benches, bench_libfuncs);
//criterion_main!(benches);
criterion_main!(benches);
