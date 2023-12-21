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

    let aot_factorial =
        aot_cache.compile_and_insert(Felt::from(0), &factorial, OptLevel::Aggressive);
    let aot_fibonacci =
        aot_cache.compile_and_insert(Felt::from(1), &fibonacci, OptLevel::Aggressive);
    let aot_logistic_map =
        aot_cache.compile_and_insert(Felt::from(2), &logistic_map, OptLevel::Aggressive);

    let jit_factorial =
        jit_cache.compile_and_insert(Felt::from(0), &factorial, OptLevel::Aggressive);
    let jit_fibonacci =
        jit_cache.compile_and_insert(Felt::from(1), &fibonacci, OptLevel::Aggressive);
    let jit_logistic_map =
        jit_cache.compile_and_insert(Felt::from(2), &logistic_map, OptLevel::Aggressive);

    let factorial_function_id = find_function_id(&factorial, "factorial_2M::factorial_2M::main");
    let fibonacci_function_id = find_function_id(&fibonacci, "fib_2M::fib_2M::main");
    let logistic_map_function_id =
        find_function_id(&logistic_map, "logistic_map::logistic_map::main");

    c.bench_function("Cached JIT factorial_2M", |b| {
        b.iter(|| jit_factorial.invoke_dynamic(factorial_function_id, &[], Some(u128::MAX), None));
    });
    c.bench_function("Cached JIT fib_2M", |b| {
        b.iter(|| jit_fibonacci.invoke_dynamic(fibonacci_function_id, &[], Some(u128::MAX), None));
    });
    c.bench_function("Cached JIT logistic_map", |b| {
        b.iter(|| {
            jit_logistic_map.invoke_dynamic(logistic_map_function_id, &[], Some(u128::MAX), None)
        });
    });

    c.bench_function("Cached AOT factorial_2M", |b| {
        b.iter(|| aot_factorial.invoke_dynamic(factorial_function_id, &[], Some(u128::MAX), None));
    });
    c.bench_function("Cached AOT fib_2M", |b| {
        b.iter(|| aot_fibonacci.invoke_dynamic(fibonacci_function_id, &[], Some(u128::MAX), None));
    });
    c.bench_function("Cached AOT logistic_map", |b| {
        b.iter(|| {
            aot_logistic_map.invoke_dynamic(logistic_map_function_id, &[], Some(u128::MAX), None)
        });
    });

    unsafe {
        let aot_factorial_fn = std::mem::transmute::<*const (), extern "C" fn(u128)>(
            aot_factorial
                .find_function_ptr(factorial_function_id)
                .cast(),
        );
        c.bench_function("Cached AOT factorial_2M (direct invoke)", |b| {
            b.iter(|| aot_factorial_fn(u128::MAX));
        });

        // let aot_factorial_fn: &extern "C" fn(u128) =
        //     std::mem::transmute(&*aot_factorial.find_function_ptr(factorial_function_id));
        // let aot_fibonacci_fn: extern "C" fn(u128) =
        //     std::mem::transmute_copy(&*aot_fibonacci.find_function_ptr(fibonacci_function_id));
        // let aot_logistic_map_fn: extern "C" fn(u128) = std::mem::transmute_copy(
        //     &*aot_logistic_map.find_function_ptr(logistic_map_function_id),
        // );
        // let jit_factorial_fn: extern "C" fn(u128) =
        //     std::mem::transmute_copy(&*jit_factorial.find_function_ptr(factorial_function_id));
        // let jit_fibonacci_fn: extern "C" fn(u128) =
        //     std::mem::transmute_copy(&*jit_fibonacci.find_function_ptr(fibonacci_function_id));
        // let jit_logistic_map_fn: extern "C" fn(u128) = std::mem::transmute_copy(
        //     &*jit_logistic_map.find_function_ptr(logistic_map_function_id),
        // );

        // c.bench_function("Cached JIT factorial_2M (direct invoke)", |b| {
        //     b.iter(|| jit_factorial_fn(u128::MAX));
        // });
        // c.bench_function("Cached JIT fib_2M (direct invoke)", |b| {
        //     b.iter(|| jit_fibonacci_fn(u128::MAX));
        // });
        // c.bench_function("Cached JIT logistic_map (direct invoke)", |b| {
        //     b.iter(|| jit_logistic_map_fn(u128::MAX));
        // });

        // c.bench_function("Cached AOT factorial_2M (direct invoke)", |b| {
        //     b.iter(|| aot_factorial_fn(u128::MAX));
        // });
        // c.bench_function("Cached AOT fib_2M (direct invoke)", |b| {
        //     b.iter(|| aot_fibonacci_fn(u128::MAX));
        // });
        // c.bench_function("Cached AOT logistic_map (direct invoke)", |b| {
        //     b.iter(|| aot_logistic_map_fn(u128::MAX));
        // });
    }
}

fn load_contract(path: impl AsRef<Path>) -> Program {
    let mut db = RootDatabase::builder().detect_corelib().build().unwrap();
    let main_crate_ids = setup_project(&mut db, path.as_ref()).unwrap();
    (*compile_prepared_db(
        &mut db,
        main_crate_ids,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap())
    .clone()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
