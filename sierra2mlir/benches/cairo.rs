use cairo_lang_runner::{SierraCasmRunner, StarknetState};
use cairo_lang_sierra::ProgramParser;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn benchmark_cairo(c: &mut Criterion) {
    let program = ProgramParser::new().parse(include_str!("programs/fib.sierra")).unwrap();
    let runner =
        SierraCasmRunner::new(program, Some(Default::default()), Default::default()).unwrap();
    let starknet_state: StarknetState = Default::default();

    let func = runner.find_function("::main").unwrap().clone();

    c.bench_with_input(
        BenchmarkId::new("Cairo", 1),
        &(runner, func, starknet_state),
        |b, (runner, func, starknet_state)| {
            b.iter(|| runner.run_function(func, &[], Some(800_000_000), starknet_state.clone()))
        },
    );
}

criterion_group!(benches, benchmark_cairo);
criterion_main!(benches);
