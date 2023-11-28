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
use cairo_lang_starknet::{
    contract_class::compile_contract_in_prepared_db, inline_macros::selector::SelectorMacro,
    plugin::StarkNetPlugin,
};
use cairo_native::{
    debug_info::{DebugInfo, DebugLocations},
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
};
use clap::Parser;
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_llvm_translations},
    Context,
};
use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input .sierra or .cairo program.
    input: PathBuf,

    /// Output file, .so on linux, .dylib on macOS
    output: PathBuf,

    /// Whether the program is a contract.
    #[arg(short, long)]
    starknet: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments.
    let args = Args::parse();

    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    // Load the program.
    let context = Context::new();
    let (program, debug_info) =
        load_program(Path::new(&args.input), Some(&context), args.starknet)?;

    // Initialize MLIR.
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);

    // Compile the program.
    let mut module = Module::new(Location::unknown(&context));
    let mut metadata = MetadataStorage::new();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program)?;

    // Make the runtime library available.
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    cairo_native::compile::<CoreType, CoreLibfunc>(
        &context,
        &module,
        &program,
        &registry,
        &mut metadata,
        debug_info.as_ref(),
    )?;

    // lower to llvm dialect
    let pass_manager = PassManager::new(&context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_index_to_llvm());
    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager.run(&mut module)?;

    let object = cairo_native::module_to_object(&module)?;
    cairo_native::object_to_shared_lib(&object, &args.output)?;

    Ok(())
}

fn load_program<'c>(
    path: &Path,
    context: Option<&'c Context>,
    is_contract: bool,
) -> Result<(Program, Option<DebugLocations<'c>>), Box<dyn std::error::Error>> {
    Ok(match path.extension().and_then(OsStr::to_str) {
        Some("cairo") if !is_contract => {
            let mut db = RootDatabase::builder().detect_corelib().build()?;
            let main_crate_ids = setup_project(&mut db, path)?;
            let program = (*compile_prepared_db(
                &mut db,
                main_crate_ids,
                CompilerConfig {
                    replace_ids: true,
                    ..Default::default()
                },
            )?)
            .clone();

            let debug_locations = if let Some(context) = context {
                let debug_info = DebugInfo::extract(&db, &program).map_err(|_| {
                    let mut buffer = String::new();
                    assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
                    buffer
                })?;

                Some(DebugLocations::extract(context, &db, &debug_info))
            } else {
                None
            };

            (program, debug_locations)
        }
        Some("cairo") if is_contract => {
            // mimics cairo_lang_starknet::contract_class::compile_path
            let mut db = RootDatabase::builder()
                .detect_corelib()
                .with_macro_plugin(Arc::new(StarkNetPlugin::default()))
                .with_inline_macro_plugin(SelectorMacro::NAME, Arc::new(SelectorMacro))
                .build()?;

            let main_crate_ids = setup_project(&mut db, Path::new(&path))?;

            let contract = compile_contract_in_prepared_db(
                &db,
                None,
                main_crate_ids,
                CompilerConfig {
                    replace_ids: true,
                    ..Default::default()
                },
            )?;

            let program = contract.extract_sierra_program()?;

            let debug_locations = if let Some(context) = context {
                let debug_info = DebugInfo::extract(&db, &program).map_err(|_| {
                    let mut buffer = String::new();
                    assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
                    buffer
                })?;

                Some(DebugLocations::extract(context, &db, &debug_info))
            } else {
                None
            };

            (program, debug_locations)
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

    /// Compile a starknet contract
    #[clap(long)]
    starknet: bool,
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
        CompilerOutput::Path(PathBuf::from(input))
    })
}
