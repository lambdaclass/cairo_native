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

    let aot_factorial = aot_cache.compile_and_insert(Felt::ZERO, &factorial, OptLevel::None);
    let aot_fibonacci = aot_cache.compile_and_insert(Felt::ONE, &fibonacci, OptLevel::None);
    let aot_logistic_map =
        aot_cache.compile_and_insert(Felt::from(2), &logistic_map, OptLevel::None);

    let jit_factorial = jit_cache.compile_and_insert(Felt::ZERO, &factorial, OptLevel::None);
    let jit_fibonacci = jit_cache.compile_and_insert(Felt::ONE, &fibonacci, OptLevel::None);
    let jit_logistic_map =
        jit_cache.compile_and_insert(Felt::from(2), &logistic_map, OptLevel::None);

    let factorial_function_id = find_function_id(&factorial, "factorial_2M::factorial_2M::main");
    let fibonacci_function_id = find_function_id(&fibonacci, "fib_2M::fib_2M::main");
    let logistic_map_function_id =
        find_function_id(&logistic_map, "logistic_map::logistic_map::main");

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

    #[cfg(target_arch = "x86_64")]
    {
        use std::mem::MaybeUninit;

        #[allow(dead_code)]
        struct PanicResult {
            tag: u8,
            payload: MaybeUninit<(i32, i32, *mut [u64; 4])>,
        }

        let aot_factorial_fn = unsafe {
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
                aot_factorial
                    .find_function_ptr(factorial_function_id)
                    .cast(),
            )
        };
        let aot_fibonacci_fn = unsafe {
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
                aot_fibonacci
                    .find_function_ptr(fibonacci_function_id)
                    .cast(),
            )
        };
        let aot_logistic_map_fn = unsafe {
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
                aot_logistic_map
                    .find_function_ptr(logistic_map_function_id)
                    .cast(),
            )
        };
        let jit_factorial_fn = unsafe {
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
                jit_factorial
                    .find_function_ptr(factorial_function_id)
                    .cast(),
            )
        };
        let jit_fibonacci_fn = unsafe {
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
                jit_fibonacci
                    .find_function_ptr(fibonacci_function_id)
                    .cast(),
            )
        };
        let jit_logistic_map_fn = unsafe {
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
                jit_logistic_map
                    .find_function_ptr(logistic_map_function_id)
                    .cast(),
            )
        };

        c.bench_function("Cached JIT factorial_2M (direct invoke)", |b| {
            b.iter(|| jit_factorial_fn(u128::MAX));
        });
        c.bench_function("Cached JIT fib_2M (direct invoke)", |b| {
            b.iter(|| jit_fibonacci_fn(u128::MAX));
        });
        c.bench_function("Cached JIT logistic_map (direct invoke)", |b| {
            b.iter(|| jit_logistic_map_fn(u128::MAX));
        });

        c.bench_function("Cached AOT factorial_2M (direct invoke)", |b| {
            b.iter(|| aot_factorial_fn(u128::MAX));
        });
        c.bench_function("Cached AOT fib_2M (direct invoke)", |b| {
            b.iter(|| aot_fibonacci_fn(u128::MAX));
        });
        c.bench_function("Cached AOT logistic_map (direct invoke)", |b| {
            b.iter(|| aot_logistic_map_fn(u128::MAX));
        });
    }
}

fn load_contract(path: impl AsRef<Path>) -> Program {
    let mut db = RootDatabase::builder().detect_corelib().build().unwrap();
    let main_crate_ids = setup_project(&mut db, path.as_ref()).unwrap();
    compile_prepared_db(
        &mut db,
        main_crate_ids,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap()
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
