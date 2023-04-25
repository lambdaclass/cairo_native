use cairo_lang_sierra::ProgramParser;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{env, fs};

pub fn criterion_benchmark(c: &mut Criterion) {
    let program = ProgramParser::new().parse(include_str!("programs/fib.sierra")).unwrap();
    let engine = sierra2mlir::execute(&program, false, 1).unwrap();

    unsafe {
        engine.invoke_packed("fib::fib::main", &mut []).unwrap();
    };

    c.bench_with_input(BenchmarkId::new("MLIR", 1), &(engine), |b, engine| {
        b.iter(|| {
            unsafe {
                engine.invoke_packed("fib::fib::main", &mut []).ok();
            };
        });
    });

    // Requires sierra files to be generated previously.
    let mut compile_group = c.benchmark_group("compile");

    let base_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    for entry in fs::read_dir(format!("{base_dir}/../examples")).unwrap() {
        let entry = entry.unwrap();
        let path = fs::canonicalize(entry.path()).unwrap();

        if let Some(ext) = path.extension() {
            if ext.eq_ignore_ascii_case("sierra") {
                compile_group.bench_function(
                    &format!("examples/{}", path.file_stem().unwrap().to_string_lossy()),
                    move |x| {
                        let sierra_code = fs::read_to_string(&path).unwrap();
                        let program = ProgramParser::new().parse(&sierra_code).unwrap();
                        x.iter(|| {
                            sierra2mlir::compile(&program, false, false, false, 1).unwrap();
                        });
                    },
                );
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
