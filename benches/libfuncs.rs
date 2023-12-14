use cairo_native::{context::NativeContext, executor::NativeExecutor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use util::prepare_programs;

mod util;

pub fn bench_libfuncs(c: &mut Criterion) {
    let programs = prepare_programs("tests/cases");

    {
        let mut c = c.benchmark_group("Libfunc Cold JIT Execution Time");

        for (program, filename) in &programs {
            c.bench_with_input(BenchmarkId::new(filename, 1), &program, |b, program| {
                let native_context = NativeContext::new();
                b.iter(|| {
                    let module = native_context.compile(program).unwrap();
                    // pass manager internally verifies the MLIR output is correct.
                    let native_executor = NativeExecutor::new(module);

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

                    // Execute the program.
                    let result = native_executor
                        .execute(&entry.id, &[], Some(u64::MAX as u128))
                        .unwrap();
                    black_box(result)
                })
            });
        }
    }

    {
        let mut c = c.benchmark_group("Libfunc Hot JIT Execution Time");

        for (program, filename) in &programs {
            if filename == "div.cairo" {
                continue; // todo: enable when libfuncs felt252_div and felt252_div_const are implemented
            }
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

            c.bench_with_input(BenchmarkId::new(filename, 1), program, |b, program| {
                let native_context = NativeContext::new();
                let module = native_context.compile(program).unwrap();
                // pass manager internally verifies the MLIR output is correct.
                let native_executor = NativeExecutor::new(module);

                // warmup
                for _ in 0..5 {
                    native_executor
                        .execute(&entry.id, &[], Some(u64::MAX as u128))
                        .unwrap();
                }

                b.iter(|| {
                    // Execute the program.
                    let result = native_executor
                        .execute(&entry.id, &[], Some(u64::MAX as u128))
                        .unwrap();
                    black_box(result)
                })
            });
        }
    }
}

criterion_group!(benches, bench_libfuncs);
criterion_main!(benches);
