//use cairo_lang_compiler::{
use cairo_lang_compiler::{
//    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
//};
};
//use cairo_lang_sierra::program::Program;
use cairo_lang_sierra::program::Program;
//use cairo_native::{
use cairo_native::{
//    cache::{AotProgramCache, JitProgramCache},
    cache::{AotProgramCache, JitProgramCache},
//    context::NativeContext,
    context::NativeContext,
//    utils::find_function_id,
    utils::find_function_id,
//    OptLevel,
    OptLevel,
//};
};
//use criterion::{criterion_group, criterion_main, Criterion};
use criterion::{criterion_group, criterion_main, Criterion};
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use std::path::Path;
use std::path::Path;
//

//fn criterion_benchmark(c: &mut Criterion) {
fn criterion_benchmark(c: &mut Criterion) {
//    let context = NativeContext::new();
    let context = NativeContext::new();
//    let mut aot_cache = AotProgramCache::new(&context);
    let mut aot_cache = AotProgramCache::new(&context);
//    let mut jit_cache = JitProgramCache::new(&context);
    let mut jit_cache = JitProgramCache::new(&context);
//

//    let factorial = load_contract("programs/benches/factorial_2M.cairo");
    let factorial = load_contract("programs/benches/factorial_2M.cairo");
//    let fibonacci = load_contract("programs/benches/fib_2M.cairo");
    let fibonacci = load_contract("programs/benches/fib_2M.cairo");
//    let logistic_map = load_contract("programs/benches/logistic_map.cairo");
    let logistic_map = load_contract("programs/benches/logistic_map.cairo");
//

//    let aot_factorial = aot_cache.compile_and_insert(Felt::ZERO, &factorial, OptLevel::None);
    let aot_factorial = aot_cache.compile_and_insert(Felt::ZERO, &factorial, OptLevel::None);
//    let aot_fibonacci = aot_cache.compile_and_insert(Felt::ONE, &fibonacci, OptLevel::None);
    let aot_fibonacci = aot_cache.compile_and_insert(Felt::ONE, &fibonacci, OptLevel::None);
//    let aot_logistic_map =
    let aot_logistic_map =
//        aot_cache.compile_and_insert(Felt::from(2), &logistic_map, OptLevel::None);
        aot_cache.compile_and_insert(Felt::from(2), &logistic_map, OptLevel::None);
//

//    let jit_factorial = jit_cache.compile_and_insert(Felt::ZERO, &factorial, OptLevel::None);
    let jit_factorial = jit_cache.compile_and_insert(Felt::ZERO, &factorial, OptLevel::None);
//    let jit_fibonacci = jit_cache.compile_and_insert(Felt::ONE, &fibonacci, OptLevel::None);
    let jit_fibonacci = jit_cache.compile_and_insert(Felt::ONE, &fibonacci, OptLevel::None);
//    let jit_logistic_map =
    let jit_logistic_map =
//        jit_cache.compile_and_insert(Felt::from(2), &logistic_map, OptLevel::None);
        jit_cache.compile_and_insert(Felt::from(2), &logistic_map, OptLevel::None);
//

//    let factorial_function_id = find_function_id(&factorial, "factorial_2M::factorial_2M::main");
    let factorial_function_id = find_function_id(&factorial, "factorial_2M::factorial_2M::main");
//    let fibonacci_function_id = find_function_id(&fibonacci, "fib_2M::fib_2M::main");
    let fibonacci_function_id = find_function_id(&fibonacci, "fib_2M::fib_2M::main");
//    let logistic_map_function_id =
    let logistic_map_function_id =
//        find_function_id(&logistic_map, "logistic_map::logistic_map::main");
        find_function_id(&logistic_map, "logistic_map::logistic_map::main");
//

//    c.bench_function("Cached JIT factorial_2M", |b| {
    c.bench_function("Cached JIT factorial_2M", |b| {
//        b.iter(|| jit_factorial.invoke_dynamic(factorial_function_id, &[], Some(u128::MAX)));
        b.iter(|| jit_factorial.invoke_dynamic(factorial_function_id, &[], Some(u128::MAX)));
//    });
    });
//    c.bench_function("Cached JIT fib_2M", |b| {
    c.bench_function("Cached JIT fib_2M", |b| {
//        b.iter(|| jit_fibonacci.invoke_dynamic(fibonacci_function_id, &[], Some(u128::MAX)));
        b.iter(|| jit_fibonacci.invoke_dynamic(fibonacci_function_id, &[], Some(u128::MAX)));
//    });
    });
//    c.bench_function("Cached JIT logistic_map", |b| {
    c.bench_function("Cached JIT logistic_map", |b| {
//        b.iter(|| jit_logistic_map.invoke_dynamic(logistic_map_function_id, &[], Some(u128::MAX)));
        b.iter(|| jit_logistic_map.invoke_dynamic(logistic_map_function_id, &[], Some(u128::MAX)));
//    });
    });
//

//    c.bench_function("Cached AOT factorial_2M", |b| {
    c.bench_function("Cached AOT factorial_2M", |b| {
//        b.iter(|| aot_factorial.invoke_dynamic(factorial_function_id, &[], Some(u128::MAX)));
        b.iter(|| aot_factorial.invoke_dynamic(factorial_function_id, &[], Some(u128::MAX)));
//    });
    });
//    c.bench_function("Cached AOT fib_2M", |b| {
    c.bench_function("Cached AOT fib_2M", |b| {
//        b.iter(|| aot_fibonacci.invoke_dynamic(fibonacci_function_id, &[], Some(u128::MAX)));
        b.iter(|| aot_fibonacci.invoke_dynamic(fibonacci_function_id, &[], Some(u128::MAX)));
//    });
    });
//    c.bench_function("Cached AOT logistic_map", |b| {
    c.bench_function("Cached AOT logistic_map", |b| {
//        b.iter(|| aot_logistic_map.invoke_dynamic(logistic_map_function_id, &[], Some(u128::MAX)));
        b.iter(|| aot_logistic_map.invoke_dynamic(logistic_map_function_id, &[], Some(u128::MAX)));
//    });
    });
//

//    #[cfg(target_arch = "x86_64")]
    #[cfg(target_arch = "x86_64")]
//    {
    {
//        use std::mem::MaybeUninit;
        use std::mem::MaybeUninit;
//

//        #[allow(dead_code)]
        #[allow(dead_code)]
//        struct PanicResult {
        struct PanicResult {
//            tag: u8,
            tag: u8,
//            payload: MaybeUninit<(i32, i32, *mut [u64; 4])>,
            payload: MaybeUninit<(i32, i32, *mut [u64; 4])>,
//        }
        }
//

//        let aot_factorial_fn = unsafe {
        let aot_factorial_fn = unsafe {
//            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
//                aot_factorial
                aot_factorial
//                    .find_function_ptr(factorial_function_id)
                    .find_function_ptr(factorial_function_id)
//                    .cast(),
                    .cast(),
//            )
            )
//        };
        };
//        let aot_fibonacci_fn = unsafe {
        let aot_fibonacci_fn = unsafe {
//            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
//                aot_fibonacci
                aot_fibonacci
//                    .find_function_ptr(fibonacci_function_id)
                    .find_function_ptr(fibonacci_function_id)
//                    .cast(),
                    .cast(),
//            )
            )
//        };
        };
//        let aot_logistic_map_fn = unsafe {
        let aot_logistic_map_fn = unsafe {
//            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
//                aot_logistic_map
                aot_logistic_map
//                    .find_function_ptr(logistic_map_function_id)
                    .find_function_ptr(logistic_map_function_id)
//                    .cast(),
                    .cast(),
//            )
            )
//        };
        };
//        let jit_factorial_fn = unsafe {
        let jit_factorial_fn = unsafe {
//            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
//                jit_factorial
                jit_factorial
//                    .find_function_ptr(factorial_function_id)
                    .find_function_ptr(factorial_function_id)
//                    .cast(),
                    .cast(),
//            )
            )
//        };
        };
//        let jit_fibonacci_fn = unsafe {
        let jit_fibonacci_fn = unsafe {
//            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
//                jit_fibonacci
                jit_fibonacci
//                    .find_function_ptr(fibonacci_function_id)
                    .find_function_ptr(fibonacci_function_id)
//                    .cast(),
                    .cast(),
//            )
            )
//        };
        };
//        let jit_logistic_map_fn = unsafe {
        let jit_logistic_map_fn = unsafe {
//            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
            std::mem::transmute::<*const (), extern "C" fn(u128) -> (u128, PanicResult)>(
//                jit_logistic_map
                jit_logistic_map
//                    .find_function_ptr(logistic_map_function_id)
                    .find_function_ptr(logistic_map_function_id)
//                    .cast(),
                    .cast(),
//            )
            )
//        };
        };
//

//        c.bench_function("Cached JIT factorial_2M (direct invoke)", |b| {
        c.bench_function("Cached JIT factorial_2M (direct invoke)", |b| {
//            b.iter(|| jit_factorial_fn(u128::MAX));
            b.iter(|| jit_factorial_fn(u128::MAX));
//        });
        });
//        c.bench_function("Cached JIT fib_2M (direct invoke)", |b| {
        c.bench_function("Cached JIT fib_2M (direct invoke)", |b| {
//            b.iter(|| jit_fibonacci_fn(u128::MAX));
            b.iter(|| jit_fibonacci_fn(u128::MAX));
//        });
        });
//        c.bench_function("Cached JIT logistic_map (direct invoke)", |b| {
        c.bench_function("Cached JIT logistic_map (direct invoke)", |b| {
//            b.iter(|| jit_logistic_map_fn(u128::MAX));
            b.iter(|| jit_logistic_map_fn(u128::MAX));
//        });
        });
//

//        c.bench_function("Cached AOT factorial_2M (direct invoke)", |b| {
        c.bench_function("Cached AOT factorial_2M (direct invoke)", |b| {
//            b.iter(|| aot_factorial_fn(u128::MAX));
            b.iter(|| aot_factorial_fn(u128::MAX));
//        });
        });
//        c.bench_function("Cached AOT fib_2M (direct invoke)", |b| {
        c.bench_function("Cached AOT fib_2M (direct invoke)", |b| {
//            b.iter(|| aot_fibonacci_fn(u128::MAX));
            b.iter(|| aot_fibonacci_fn(u128::MAX));
//        });
        });
//        c.bench_function("Cached AOT logistic_map (direct invoke)", |b| {
        c.bench_function("Cached AOT logistic_map (direct invoke)", |b| {
//            b.iter(|| aot_logistic_map_fn(u128::MAX));
            b.iter(|| aot_logistic_map_fn(u128::MAX));
//        });
        });
//    }
    }
//}
}
//

//fn load_contract(path: impl AsRef<Path>) -> Program {
fn load_contract(path: impl AsRef<Path>) -> Program {
//    let mut db = RootDatabase::builder().detect_corelib().build().unwrap();
    let mut db = RootDatabase::builder().detect_corelib().build().unwrap();
//    let main_crate_ids = setup_project(&mut db, path.as_ref()).unwrap();
    let main_crate_ids = setup_project(&mut db, path.as_ref()).unwrap();
//    compile_prepared_db(
    compile_prepared_db(
//        &mut db,
        &mut db,
//        main_crate_ids,
        main_crate_ids,
//        CompilerConfig {
        CompilerConfig {
//            replace_ids: true,
            replace_ids: true,
//            ..Default::default()
            ..Default::default()
//        },
        },
//    )
    )
//    .unwrap()
    .unwrap()
//}
}
//

//criterion_group!(benches, criterion_benchmark);
criterion_group!(benches, criterion_benchmark);
//criterion_main!(benches);
criterion_main!(benches);
