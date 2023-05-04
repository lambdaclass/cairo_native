use cairo_lang_sierra::ProgramParser;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{env, fs};

// Source: the generated MLIR from fib.sierra
#[derive(Debug, Default)]
#[repr(C, packed)]
struct ReturnValue {
    pub tag: i16,
    pub data: [i8; 16],
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let program = ProgramParser::new().parse(include_str!("programs/fib.sierra")).unwrap();
    let engine = sierra2mlir::execute(&program, false, 1, Some(800_000_000)).unwrap();

    let mut return_value = ReturnValue::default();
    let mut return_ptr = std::ptr::addr_of_mut!(return_value);
    let return_ptr_ptr = std::ptr::addr_of_mut!(return_ptr);
    unsafe {
        engine.invoke_packed("fib::fib::main", &mut [return_ptr_ptr as *mut _ as *mut ()]).unwrap();
    };

    c.bench_with_input(BenchmarkId::new("MLIR", 1), &(engine), |b, engine| {
        b.iter(|| {
            unsafe {
                engine
                    .invoke_packed("fib::fib::main", &mut [return_ptr_ptr as *mut _ as *mut ()])
                    .ok();
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
                            sierra2mlir::compile(
                                &program,
                                false,
                                false,
                                false,
                                1,
                                Some(100_000_000),
                            )
                            .unwrap();
                        });
                    },
                );
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
