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

fn compare(c: &mut Criterion, path: impl AsRef<Path>) {
    let context = NativeContext::new();
    let mut aot_cache = AotProgramCache::new(&context);
    let mut jit_cache = JitProgramCache::new(&context);

    let stem = path
        .as_ref()
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let program = load_contract(&path);
    let aot_executor = aot_cache
        .compile_and_insert(Felt::ZERO, &program, OptLevel::Aggressive)
        .unwrap();
    let jit_executor = jit_cache
        .compile_and_insert(Felt::ZERO, &program, OptLevel::Aggressive)
        .unwrap();

    let main_name = format!("{stem}::{stem}::main");
    let main_id = find_function_id(&program, &main_name).unwrap();

    let vm_runner = load_contract_for_vm(&path);
    let vm_main_id = vm_runner
        .find_function("main")
        .expect("failed to find main function");

    let mut group = c.benchmark_group(stem);

    group.bench_function("Cached JIT", |b| {
        b.iter(|| {
            let result = jit_executor
                .invoke_dynamic(main_id, &[], Some(u64::MAX))
                .unwrap();
            let value = result.return_value;
            assert!(matches!(value, Value::Enum { tag: 0, .. }))
        });
    });
    group.bench_function("Cached AOT", |b| {
        b.iter(|| {
            let result = aot_executor
                .invoke_dynamic(main_id, &[], Some(u64::MAX))
                .unwrap();
            let value = result.return_value;
            assert!(matches!(value, Value::Enum { tag: 0, .. }))
        });
    });
    group.bench_function("VM", |b| {
        b.iter(|| {
            let result = vm_runner
                .run_function_with_starknet_context(
                    vm_main_id,
                    vec![],
                    Some(usize::MAX),
                    StarknetState::default(),
                )
                .unwrap();
            let value = result.value;
            assert!(matches!(value, RunResultValue::Success(_)))
        });
    });

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    compare(c, "programs/benches/dict_snapshot.cairo");
    // compare(c, "programs/benches/dict_insert.cairo");
    // compare(c, "programs/benches/factorial_2M.cairo");
    // compare(c, "programs/benches/fib_2M.cairo");
    // compare(c, "programs/benches/linear_search.cairo");
    // compare(c, "programs/benches/logistic_map.cairo");
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
