use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
};
use cairo_lang_runner::{RunResultValue, SierraCasmRunner, StarknetState};
use cairo_lang_sierra::program::Program;
use cairo_lang_sierra_generator::replace_ids::DebugReplacer;
use cairo_lang_starknet::contract::{find_contracts, get_contracts_info};
use cairo_lang_utils::Upcast;
use cairo_native::{
    cache::{AotProgramCache, JitProgramCache},
    context::NativeContext,
    utils::find_function_id,
    OptLevel, Value,
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
    let linear_search = load_contract("programs/benches/linear_search.cairo");

    let aot_factorial = aot_cache
        .compile_and_insert(Felt::ZERO, &factorial, OptLevel::Aggressive)
        .unwrap();
    let aot_fibonacci = aot_cache
        .compile_and_insert(Felt::ONE, &fibonacci, OptLevel::Aggressive)
        .unwrap();
    let aot_logistic_map = aot_cache
        .compile_and_insert(Felt::from(2), &logistic_map, OptLevel::Aggressive)
        .unwrap();
    let aot_linear_search = aot_cache
        .compile_and_insert(Felt::from(2), &linear_search, OptLevel::Aggressive)
        .unwrap();

    let jit_factorial = jit_cache
        .compile_and_insert(Felt::ZERO, &factorial, OptLevel::Aggressive)
        .unwrap();
    let jit_fibonacci = jit_cache
        .compile_and_insert(Felt::ONE, &fibonacci, OptLevel::Aggressive)
        .unwrap();
    let jit_logistic_map = jit_cache
        .compile_and_insert(Felt::from(2), &logistic_map, OptLevel::Aggressive)
        .unwrap();
    let jit_linear_search = jit_cache
        .compile_and_insert(Felt::from(2), &linear_search, OptLevel::Aggressive)
        .unwrap();

    let factorial_function_id =
        find_function_id(&factorial, "factorial_2M::factorial_2M::main").unwrap();
    let fibonacci_function_id = find_function_id(&fibonacci, "fib_2M::fib_2M::main").unwrap();
    let logistic_map_function_id =
        find_function_id(&logistic_map, "logistic_map::logistic_map::main").unwrap();
    let linear_search_function_id =
        find_function_id(&linear_search, "linear_search::linear_search::main").unwrap();

    let factorial_runner = load_contract_for_vm("programs/benches/factorial_2M.cairo");
    let fibonacci_runner = load_contract_for_vm("programs/benches/fib_2M.cairo");
    let logistic_map_runner = load_contract_for_vm("programs/benches/logistic_map.cairo");
    let linear_search_runner = load_contract_for_vm("programs/benches/linear_search.cairo");

    let factorial_function = factorial_runner
        .find_function("main")
        .expect("failed to find main factorial function");
    let fibonacci_function = fibonacci_runner
        .find_function("main")
        .expect("failed to find main fibonacci function");
    let logistic_map_function = logistic_map_runner
        .find_function("main")
        .expect("failed to find main logistic map function");
    let linear_search_function = linear_search_runner
        .find_function("main")
        .expect("failed to find main logistic map function");

    {
        let mut linear_search_group = c.benchmark_group("linear_search");

        linear_search_group.bench_function("Cached JIT", |b| {
            b.iter(|| {
                let result = jit_linear_search
                    .invoke_dynamic(linear_search_function_id, &[], Some(u64::MAX))
                    .unwrap();
                let value = result.return_value;
                assert!(matches!(value, Value::Enum { tag: 0, .. }))
            });
        });
        linear_search_group.bench_function("Cached AOT", |b| {
            b.iter(|| {
                let result = aot_linear_search
                    .invoke_dynamic(linear_search_function_id, &[], Some(u64::MAX))
                    .unwrap();
                let value = result.return_value;
                assert!(matches!(value, Value::Enum { tag: 0, .. }))
            });
        });

        linear_search_group.bench_function("VM", |b| {
            b.iter(|| {
                let result = linear_search_runner
                    .run_function_with_starknet_context(
                        linear_search_function,
                        &[],
                        Some(usize::MAX),
                        StarknetState::default(),
                    )
                    .unwrap();
                let value = result.value;
                assert!(matches!(value, RunResultValue::Success(_)))
            });
        });

        linear_search_group.finish();
    }

    {
        let mut factorial_group = c.benchmark_group("factorial_2M");

        factorial_group.bench_function("Cached JIT", |b| {
            b.iter(|| {
                let result = jit_factorial
                    .invoke_dynamic(factorial_function_id, &[], Some(u64::MAX))
                    .unwrap();
                let value = result.return_value;
                assert!(matches!(value, Value::Enum { tag: 0, .. }))
            });
        });
        factorial_group.bench_function("Cached AOT", |b| {
            b.iter(|| {
                let result = aot_factorial
                    .invoke_dynamic(factorial_function_id, &[], Some(u64::MAX))
                    .unwrap();
                let value = result.return_value;
                assert!(matches!(value, Value::Enum { tag: 0, .. }))
            });
        });

        factorial_group.bench_function("VM", |b| {
            b.iter(|| {
                let result = factorial_runner
                    .run_function_with_starknet_context(
                        factorial_function,
                        &[],
                        Some(usize::MAX),
                        StarknetState::default(),
                    )
                    .unwrap();
                let value = result.value;
                assert!(matches!(value, RunResultValue::Success(_)))
            });
        });

        factorial_group.finish();
    }

    {
        let mut fibonacci_group = c.benchmark_group("fibonacci_2M");

        fibonacci_group.bench_function("Cached JIT", |b| {
            b.iter(|| {
                let result = jit_fibonacci
                    .invoke_dynamic(fibonacci_function_id, &[], Some(u64::MAX))
                    .unwrap();
                let value = result.return_value;
                assert!(matches!(value, Value::Enum { tag: 0, .. }))
            });
        });
        fibonacci_group.bench_function("Cached AOT", |b| {
            b.iter(|| {
                let result = aot_fibonacci
                    .invoke_dynamic(fibonacci_function_id, &[], Some(u64::MAX))
                    .unwrap();
                let value = result.return_value;
                assert!(matches!(value, Value::Enum { tag: 0, .. }))
            })
        });
        fibonacci_group.bench_function("VM", |b| {
            b.iter(|| {
                let result = fibonacci_runner
                    .run_function_with_starknet_context(
                        fibonacci_function,
                        &[],
                        Some(usize::MAX),
                        StarknetState::default(),
                    )
                    .unwrap();
                let value = result.value;
                assert!(matches!(value, RunResultValue::Success(_)))
            });
        });

        fibonacci_group.finish();
    }

    {
        let mut logistic_map_group = c.benchmark_group("logistic_map");

        logistic_map_group.bench_function("Cached JIT", |b| {
            b.iter(|| {
                let result = jit_logistic_map
                    .invoke_dynamic(logistic_map_function_id, &[], Some(u64::MAX))
                    .unwrap();
                let value = result.return_value;
                assert!(matches!(value, Value::Enum { tag: 0, .. }))
            });
        });

        logistic_map_group.bench_function("Cached AOT", |b| {
            b.iter(|| {
                let result = aot_logistic_map
                    .invoke_dynamic(logistic_map_function_id, &[], Some(u64::MAX))
                    .unwrap();
                let value = result.return_value;
                assert!(matches!(value, Value::Enum { tag: 0, .. }))
            });
        });

        logistic_map_group.bench_function("VM", |b| {
            b.iter(|| {
                let result = logistic_map_runner
                    .run_function_with_starknet_context(
                        logistic_map_function,
                        &[],
                        Some(usize::MAX),
                        StarknetState::default(),
                    )
                    .unwrap();
                let value = result.value;
                assert!(matches!(value, RunResultValue::Success(_)))
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
        &db,
        main_crate_ids.clone(),
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .expect("failed to compile program");

    let replacer = DebugReplacer { db: &db };
    let contracts = find_contracts((db).upcast(), &main_crate_ids);
    let contracts_info =
        get_contracts_info(&db, contracts, &replacer).expect("failed to get contracts info");

    SierraCasmRunner::new(
        program.program.clone(),
        Some(Default::default()),
        contracts_info,
        None,
    )
    .expect("failed to create runner")
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
