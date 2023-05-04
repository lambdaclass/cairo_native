use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::{program::Program, ProgramParser};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{env, fs, path::Path, sync::Arc};

// Source: the generated MLIR from fib.sierra
#[derive(Debug, Default)]
#[repr(C, packed)]
struct ReturnValue {
    pub tag: i16,
    pub data: [i8; 16],
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let program = compile_sierra_program("fib");
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
