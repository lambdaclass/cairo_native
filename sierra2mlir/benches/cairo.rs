use std::{fs, path::Path, sync::Arc};

use cairo_lang_compiler::CompilerConfig;
use cairo_lang_runner::{SierraCasmRunner, StarknetState};
use cairo_lang_sierra::{program::Program, ProgramParser};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn benchmark_cairo(c: &mut Criterion) {
    let program = compile_sierra_program("fib");
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

fn compile_sierra_program(program_name: &str) -> Program {
    let test_path = Path::new(".").join("benches").join("programs").join(program_name);
    let sierra_path = test_path.with_extension("sierra");
    let cairo_path = test_path.with_extension("cairo");

    if sierra_path.exists() {
        let sierra_code =
            fs::read_to_string(format!("./benches/programs/{program_name}.sierra")).unwrap();
        ProgramParser::new().parse(&sierra_code).unwrap()
    } else if cairo_path.exists() {
        let program_ptr = cairo_lang_compiler::compile_cairo_project_at_path(
            &cairo_path,
            CompilerConfig { replace_ids: true, ..Default::default() },
        )
        .expect("Cairo compilation failed");
        let program = Arc::try_unwrap(program_ptr).unwrap();
        fs::write(sierra_path, program.to_string()).unwrap();
        program
    } else {
        panic!("Cannot find {program_name}.sierra or {program_name}.cairo")
    }
}
