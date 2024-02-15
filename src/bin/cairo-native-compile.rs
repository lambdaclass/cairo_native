use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
    project::setup_project, CompilerConfig,
};
use cairo_lang_defs::plugin::NamedPlugin;
use cairo_lang_semantic::plugin::PluginSuite;
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
    utils::run_pass_manager,
    OptLevel,
};
use clap::Parser;
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
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

    cairo_native::compile(
        &context,
        &module,
        &program,
        &registry,
        &mut metadata,
        debug_info.as_ref(),
    )?;

    run_pass_manager(&context, &mut module)?;

    let opt_level = match args.opt_level {
        0 => OptLevel::None,
        1 => OptLevel::Less,
        2 => OptLevel::Default,
        _ => OptLevel::Aggressive,
    };

    let object = cairo_native::module_to_object(&module, opt_level)?;
    cairo_native::object_to_shared_lib(
        &object,
        match &args.output {
            CompilerOutput::Stdout => Path::new("/dev/stdout"),
            CompilerOutput::Path(x) => x,
        },
    )?;

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
            let program = compile_prepared_db(
                &mut db,
                main_crate_ids,
                CompilerConfig {
                    replace_ids: true,
                    ..Default::default()
                },
            )?;

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
            let mut plugins = PluginSuite::default();
            plugins
                .add_plugin::<StarkNetPlugin>()
                .add_inline_macro_plugin_ex(SelectorMacro::NAME, Arc::new(SelectorMacro));
            let mut db = RootDatabase::builder()
                .detect_corelib()
                .with_plugin_suite(plugins)
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

    /// Note: This is bugged for any non-zero values. See https://github.com/lambdaclass/cairo_native/issues/404.
    #[clap(short = 'O', long, default_value = "0")]
    opt_level: usize,

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
