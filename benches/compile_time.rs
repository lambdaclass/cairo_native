use cairo_native::{context::NativeContext, module_to_object};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use util::prepare_programs;

mod util;

pub fn bench_compile_time(c: &mut Criterion) {
    {
        let mut c = c.benchmark_group("Compilation With Independent Context");

        let programs = prepare_programs();

        for (program, filename) in programs {
            c.bench_with_input(BenchmarkId::new(filename, 1), &program, |b, program| {
                b.iter(|| {
                    let native_context = NativeContext::new();
                    native_context.compile(program).unwrap();
                    // pass manager internally verifies its correct MLIR output.
                })
            });
        }
    }

    {
        let mut c = c.benchmark_group("Compilation With Shared Context");

        let programs = prepare_programs();

        let native_context = NativeContext::new();

        for (program, filename) in programs {
            c.bench_with_input(BenchmarkId::new(filename, 1), &program, |b, program| {
                b.iter(|| {
                    native_context.compile(program).unwrap();
                    // pass manager internally verifies its correct MLIR output.
                })
            });
        }
    }

    {
        let mut c = c.benchmark_group("Compilation With Independent Context To Object Code");

        let programs = prepare_programs();

        for (program, filename) in programs {
            c.bench_with_input(BenchmarkId::new(filename, 1), &program, |b, program| {
                b.iter(|| {
                    let native_context = NativeContext::new();
                    let module = native_context.compile(black_box(program)).unwrap();
                    let object = module_to_object(module.module()).expect("to work");
                    black_box(object)
                })
            });
        }
    }

    {
        let mut c = c.benchmark_group("Compilation With Shared Context To Object Code");

        let programs = prepare_programs();

        let native_context = NativeContext::new();

        for (program, filename) in programs {
            c.bench_with_input(BenchmarkId::new(filename, 1), &program, |b, program| {
                b.iter(|| {
                    let module = native_context.compile(black_box(program)).unwrap();
                    let object = module_to_object(module.module())
                        .expect("to compile correctly to a object file");
                    black_box(object)
                })
            });
        }
    }
}

criterion_group!(benches, bench_compile_time);
criterion_main!(benches);
