use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use melior_next::{pass, utility::register_all_passes, ExecutionEngine};
use sierra2mlir::compiler::Compiler;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut compiler = Compiler::new("", None).unwrap();
    compiler.compile_hardcoded_fib().unwrap();

    let pass_manager = pass::Manager::new(&compiler.context);
    register_all_passes();
    // adding the inliner pass adds a substanstial slowdown.
    // pass_manager.add_pass(pass::transform::inliner());
    pass_manager.add_pass(pass::transform::symbol_dce());
    pass_manager.add_pass(pass::transform::cse());
    pass_manager.add_pass(pass::transform::sccp());
    pass_manager.add_pass(pass::transform::canonicalizer());
    pass_manager.add_pass(pass::conversion::convert_scf_to_cf());
    pass_manager.add_pass(pass::conversion::convert_cf_to_llvm());
    pass_manager.add_pass(pass::conversion::convert_func_to_llvm());
    pass_manager.add_pass(pass::conversion::convert_arithmetic_to_llvm());

    pass_manager.enable_verifier(true);
    pass_manager.run(&mut compiler.module).unwrap();

    let engine = ExecutionEngine::new(&compiler.module, 2, &[], false);

    let mut result: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()])
            .unwrap();
    };

    c.bench_with_input(BenchmarkId::new("MLIR", 1), &(engine), |b, engine| {
        b.iter(|| {
            let mut result: i32 = -1;
            unsafe {
                engine
                    .invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()])
                    .ok();
                black_box(result)
            };
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
