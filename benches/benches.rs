use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
};
use cairo_lang_runner::{SierraCasmRunner, StarknetState};
use cairo_lang_sierra::program::Program;
use cairo_lang_sierra_generator::replace_ids::DebugReplacer;
use cairo_lang_starknet::contract::get_contracts_info;
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

    let factorial_runner = load_contract_for_vm("programs/benches/factorial_2M.cairo");
    let fibonacci_runner = load_contract_for_vm("programs/benches/fib_2M.cairo");
    let logistic_map_runner = load_contract_for_vm("programs/benches/logistic_map.cairo");

    let factorial_function = factorial_runner
        .find_function("main")
        .expect("failed to find main factorial function");
    let fibonacci_function = fibonacci_runner
        .find_function("main")
        .expect("failed to find main fibonacci function");
    let logistic_map_function = logistic_map_runner
        .find_function("main")
        .expect("failed to find main logistic map function");

    {
        let mut factorial_group = c.benchmark_group("factorial_2M");

        factorial_group.bench_function("Cached JIT", |b| {
            b.iter(|| jit_factorial.invoke_dynamic(factorial_function_id, &[], Some(u128::MAX)));
        });
        factorial_group.bench_function("Cached AOT", |b| {
            b.iter(|| aot_factorial.invoke_dynamic(factorial_function_id, &[], Some(u128::MAX)));
        });

        factorial_group.bench_function("VM", |b| {
            b.iter(|| {
                factorial_runner.run_function_with_starknet_context(
                    factorial_function,
                    &[],
                    Some(usize::MAX),
                    StarknetState::default(),
                )
            });
        });

        factorial_group.finish();
    }

    {
        let mut fibonacci_group = c.benchmark_group("fibonacci_2M");

        fibonacci_group.bench_function("Cached JIT", |b| {
            b.iter(|| jit_fibonacci.invoke_dynamic(fibonacci_function_id, &[], Some(u128::MAX)));
        });
        fibonacci_group.bench_function("Cached AOT", |b| {
            b.iter(|| aot_fibonacci.invoke_dynamic(fibonacci_function_id, &[], Some(u128::MAX)));
        });
        fibonacci_group.bench_function("VM", |b| {
            b.iter(|| {
                fibonacci_runner.run_function_with_starknet_context(
                    fibonacci_function,
                    &[],
                    Some(usize::MAX),
                    StarknetState::default(),
                )
            });
        });

        fibonacci_group.finish();
    }

    {
        let mut logistic_map_group = c.benchmark_group("logistic_map");

        logistic_map_group.bench_function("Cached JIT", |b| {
            b.iter(|| {
                jit_logistic_map.invoke_dynamic(logistic_map_function_id, &[], Some(u128::MAX))
            });
        });

        logistic_map_group.bench_function("Cached AOT", |b| {
            b.iter(|| {
                aot_logistic_map.invoke_dynamic(logistic_map_function_id, &[], Some(u128::MAX))
            });
        });

        logistic_map_group.bench_function("VM", |b| {
            b.iter(|| {
                logistic_map_runner.run_function_with_starknet_context(
                    logistic_map_function,
                    &[],
                    Some(usize::MAX),
                    StarknetState::default(),
                )
            });
        });

        logistic_map_group.finish();
    }
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

fn load_contract_for_vm(path: impl AsRef<Path>) -> SierraCasmRunner {
    let mut db = RootDatabase::builder()
        .detect_corelib()
        .build()
        .expect("failed to build database");
    let main_crate_ids = setup_project(&mut db, path.as_ref()).expect("failed to setup project");
    let program = compile_prepared_db(
        &mut db,
        main_crate_ids.clone(),
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .expect("failed to compile program");

    let replacer = DebugReplacer { db: &db };
    let contracts_info =
        get_contracts_info(&db, main_crate_ids, &replacer).expect("failed to get contracts info");

    SierraCasmRunner::new(
        program.clone(),
        Some(Default::default()),
        contracts_info,
        None,
    )
    .expect("failed to create runner")
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
