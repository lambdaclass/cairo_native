use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
};
use cairo_lang_sierra::program::Program;
use cairo_native::{
    cache::{AotProgramCache, JitProgramCache},
    context::NativeContext,
    utils::find_function_id,
    OptLevel,
};
use criterion::{criterion_group, criterion_main, Criterion};
use starknet_types_core::felt::Felt;
use std::path::Path;

fn criterion_benchmark(c: &mut Criterion) {
    let context = NativeContext::new();
    let mut aot_cache = AotProgramCache::new(&context);
    let mut jit_cache = JitProgramCache::new(&context);

    let factorial = load_contract("programs/benches/factorial_2M.cairo");
    let fibonacci = load_contract("programs/benches/fib_2M.cairo");
    let logistic_map = load_contract("programs/benches/logistic_map.cairo");

    let aot_factorial = aot_cache.compile_and_insert(Felt::ZERO, &factorial, OptLevel::Aggressive);
    let aot_fibonacci = aot_cache.compile_and_insert(Felt::ONE, &fibonacci, OptLevel::Aggressive);
    let aot_logistic_map =
        aot_cache.compile_and_insert(Felt::from(2), &logistic_map, OptLevel::Aggressive);

    let jit_factorial = jit_cache.compile_and_insert(Felt::ZERO, &factorial, OptLevel::Aggressive);
    let jit_fibonacci = jit_cache.compile_and_insert(Felt::ONE, &fibonacci, OptLevel::Aggressive);
    let jit_logistic_map =
        jit_cache.compile_and_insert(Felt::from(2), &logistic_map, OptLevel::Aggressive);

    let factorial_function_id =
        find_function_id(&factorial, "factorial_2M::factorial_2M::main").unwrap();
    let fibonacci_function_id = find_function_id(&fibonacci, "fib_2M::fib_2M::main").unwrap();
    let logistic_map_function_id =
        find_function_id(&logistic_map, "logistic_map::logistic_map::main").unwrap();

    c.bench_function("Cached JIT factorial_2M", |b| {
        b.iter(|| jit_factorial.invoke_dynamic(factorial_function_id, &[], Some(u128::MAX)));
    });
    c.bench_function("Cached JIT fib_2M", |b| {
        b.iter(|| jit_fibonacci.invoke_dynamic(fibonacci_function_id, &[], Some(u128::MAX)));
    });
    c.bench_function("Cached JIT logistic_map", |b| {
        b.iter(|| jit_logistic_map.invoke_dynamic(logistic_map_function_id, &[], Some(u128::MAX)));
    });

    c.bench_function("Cached AOT factorial_2M", |b| {
        b.iter(|| aot_factorial.invoke_dynamic(factorial_function_id, &[], Some(u128::MAX)));
    });
    c.bench_function("Cached AOT fib_2M", |b| {
        b.iter(|| aot_fibonacci.invoke_dynamic(fibonacci_function_id, &[], Some(u128::MAX)));
    });
    c.bench_function("Cached AOT logistic_map", |b| {
        b.iter(|| aot_logistic_map.invoke_dynamic(logistic_map_function_id, &[], Some(u128::MAX)));
    });
}

fn load_contract(path: impl AsRef<Path>) -> Program {
    let mut db = RootDatabase::builder().detect_corelib().build().unwrap();
    let main_crate_ids = setup_project(&mut db, path.as_ref()).unwrap();
    let sirrra_program = compile_prepared_db(
        &db,
        main_crate_ids,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    sirrra_program.program
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
