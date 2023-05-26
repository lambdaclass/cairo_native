use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::alloc::{alloc, dealloc, Layout};
use util::prepare_programs;

mod util;

// Return value layout with sensible defaults (1 felt252, aligned to 8 bytes).
const LAYOUT: Layout = unsafe { Layout::from_size_align_unchecked(32, 8) };

fn benchmark_sierra2mlir(c: &mut Criterion) {
    let programs = prepare_programs().collect::<Vec<_>>();

    // Compilation benchmarks.
    {
        let mut c = c.benchmark_group("Compilation");

        for (program, main_path) in &programs {
            c.bench_with_input(BenchmarkId::new(main_path, 1), &program, |b, program| {
                b.iter(|| {
                    sierra2mlir::compile(program, false, false, false, 1, Some(usize::MAX)).unwrap()
                });
            });
        }
    }

    // Execution benchmarks.
    {
        let mut return_mem = unsafe { alloc(LAYOUT) };

        let mut c = c.benchmark_group("Execution");
        for (program, main_path) in prepare_programs() {
            let engine = sierra2mlir::execute(&program, false, 1, Some(usize::MAX)).unwrap();

            c.bench_with_input(BenchmarkId::new(&main_path, 1), &engine, |b, engine| {
                b.iter(|| unsafe {
                    engine
                        .invoke_packed(&main_path, &mut [(&mut return_mem) as *mut _ as *mut ()])
                        .unwrap()
                });
            });
        }

        unsafe {
            dealloc(return_mem, LAYOUT);
        }
    }
}

criterion_group!(benches, benchmark_sierra2mlir);
criterion_main!(benches);
