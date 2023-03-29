use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    let engine = sierra2mlir::execute(include_str!("programs/fib.sierra"), None).unwrap();

    unsafe {
        engine.invoke_packed("fib_fib_main", &mut []).unwrap();
    };

    c.bench_with_input(BenchmarkId::new("MLIR", 1), &(engine), |b, engine| {
        b.iter(|| {
            unsafe {
                engine.invoke_packed("fib_fib_main", &mut []).ok();
            };
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
