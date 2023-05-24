use cairo_lang_runner::{SierraCasmRunner, StarknetState};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use util::prepare_programs;

mod util;

fn benchmark_cairo(c: &mut Criterion) {
    let mut c = c.benchmark_group("Interpreted");

    for (program, main_path) in prepare_programs() {
        let runner =
            SierraCasmRunner::new(program, Some(Default::default()), Default::default()).unwrap();
        let main_fn = runner.find_function(&main_path).unwrap().clone();

        let starknet_state = StarknetState::default();
        c.bench_with_input(
            BenchmarkId::new(&main_path, 1),
            &(runner, main_fn, starknet_state),
            |b, (runner, main_fn, starknet_state)| {
                b.iter(|| {
                    runner.run_function(main_fn, &[], Some(usize::MAX), starknet_state.clone())
                });
            },
        );
    }
}

criterion_group!(benches, benchmark_cairo);
criterion_main!(benches);
