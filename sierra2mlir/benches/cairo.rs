use cairo_vm::{
    cairo_run::{cairo_run, CairoRunConfig},
    hint_processor::builtin_hint_processor::builtin_hint_processor_definition::BuiltinHintProcessor,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::fs;

fn benchmark_cairo(c: &mut Criterion) {
    let cairo_run_config = CairoRunConfig::default();
    let program =
        fs::read("benches/programs/fib0.json").expect("JSON program not found or not readable");

    c.bench_with_input(
        BenchmarkId::new("Cairo", 1),
        &(program, cairo_run_config),
        |b, (program, cairo_run_config)| {
            b.iter(|| {
                let mut hint_executor = BuiltinHintProcessor::new_empty();
                cairo_run(program, cairo_run_config, &mut hint_executor)
                    .expect("Error running the Cairo program");
            })
        },
    );
}

criterion_group!(benches, benchmark_cairo);
criterion_main!(benches);
