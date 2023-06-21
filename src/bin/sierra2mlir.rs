#![feature(arc_unwrap_or_clone)]

use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
    project::setup_project, CompilerConfig,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
    ProgramParser,
};
use clap::Parser;
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    utility::register_all_dialects,
    Context,
};
use sierra2mlir::{metadata::MetadataStorage, DebugInfo};
use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments.
    let args = CmdLine::parse();

    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    // Load the program.
    let (program, _debug_info) = load_program(Path::new(&args.input))?;

    // Initialize MLIR.
    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();

    // Compile the program.
    let module = Module::new(Location::unknown(&context));
    let mut metadata = MetadataStorage::new();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program)?;

    sierra2mlir::compile::<CoreType, CoreLibfunc>(
        &context,
        &module,
        &program,
        &registry,
        &mut metadata,
    )?;

    // Write the output.
    match args.output {
        CompilerOutput::Stdout => println!("{}", module.as_operation()),
        CompilerOutput::Path(path) => fs::write(path, module.as_operation().to_string())?,
    }

    Ok(())
}

fn load_program(path: &Path) -> Result<(Program, Option<DebugInfo>), Box<dyn std::error::Error>> {
    Ok(match path.extension().and_then(OsStr::to_str) {
        Some("cairo") => {
            let mut db = RootDatabase::builder().detect_corelib().build()?;
            let main_crate_ids = setup_project(&mut db, path)?;
            let program = Arc::unwrap_or_clone(compile_prepared_db(
                &mut db,
                main_crate_ids,
                CompilerConfig {
                    replace_ids: true,
                    ..Default::default()
                },
            )?);

            let debug_info = DebugInfo::extract(&db, &program).map_err(|_| {
                let mut buffer = String::new();
                assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
                buffer
            })?;

            (program, Some(debug_info))
        }
        Some("sierra") => {
            let program_src = fs::read_to_string(path)?;

            let program_parser = ProgramParser::new();
            let program = program_parser
                .parse(&program_src)
                .map_err(|e| e.map_token(|t| t.to_string()))?;

            (program, None)
        }
        _ => unreachable!(),
    })
}

#[derive(Clone, Debug, Parser)]
struct CmdLine {
    #[clap(value_parser = parse_input)]
    input: PathBuf,

    #[clap(short = 'o', long = "output", value_parser = parse_output, default_value = "-")]
    output: CompilerOutput,
}

#[derive(Clone, Debug)]
enum CompilerOutput {
    Stdout,
    Path(PathBuf),
}

fn parse_input(input: &str) -> Result<PathBuf, String> {
    Ok(match Path::new(input).extension().and_then(OsStr::to_str) {
        Some("cairo" | "sierra") => input.into(),
        _ => {
            return Err(
                "Input path expected to have either `cairo` or `sierra` as its extension."
                    .to_string(),
            )
        }
    })
}

fn parse_output(input: &str) -> Result<CompilerOutput, String> {
    Ok(if input == "-" {
        CompilerOutput::Stdout
    } else {
        CompilerOutput::Path(match Path::new(input).extension().and_then(OsStr::to_str) {
            Some("mlir") => input.into(),
            _ => {
                return Err(
                    "Output path expected to be `-` for stdout or have `mlir` as its extension."
                        .to_string(),
                )
            }
        })
    })
}
