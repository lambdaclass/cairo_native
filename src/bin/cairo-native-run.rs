use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::FunctionId,
    program::Program,
    program_registry::ProgramRegistry,
    ProgramParser,
};
use cairo_native::{
    metadata::gas::{GasMetadata, MetadataComputationConfig},
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::register_runtime_symbols,
};
use clap::Parser;
use itertools::Itertools;
use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_passes},
    Context, ExecutionEngine,
};
use serde_json::de::StrRead;
use std::{
    borrow::Cow,
    convert::Infallible,
    ffi::OsStr,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};
use tracing::info;
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
    let program = load_program(Path::new(&args.input))?;

    let entry_point = match program
        .funcs
        .iter()
        .find(|x| x.id.debug_name == args.entry_point.debug_name || x.id == args.entry_point)
    {
        Some(x) => x,
        None => {
            // TODO: Use a proper error.
            eprintln!("Entry point `{}` not found in program.", args.entry_point);
            return Ok(());
        }
    };

    // Initialize MLIR.
    let context = Context::new();
    context.append_dialect_registry(&{
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        registry
    });
    context.load_all_available_dialects();

    register_all_passes();

    // Compile the program.
    let mut module = Module::new(Location::unknown(&context));
    let mut metadata = MetadataStorage::new();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program)?;

    // Make the runtime library available.
    metadata.insert(RuntimeBindingsMeta::default()).unwrap();

    // Gas
    let required_initial_gas = if program
        .type_declarations
        .iter()
        .any(|decl| decl.long_id.generic_id.0.as_str() == "GasBuiltin")
    {
        let gas_metadata = GasMetadata::new(&program, MetadataComputationConfig::default());

        let required_initial_gas = { gas_metadata.get_initial_required_gas(&entry_point.id) };
        info!(
            "Initial required gas: {}",
            required_initial_gas.unwrap_or(0)
        );
        // Metadata used to insert another metadata on each statement, so withdraw gas can know how much to withdraw.
        metadata.insert(gas_metadata).unwrap();
        required_initial_gas
    } else {
        None
    };

    cairo_native::compile::<CoreType, CoreLibfunc>(
        &context,
        &module,
        &program,
        &registry,
        &mut metadata,
        None,
    )?;

    // Lower to LLVM.
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

    // Create the JIT engine.
    let engine = ExecutionEngine::new(&module, 3, &[], false);

    #[cfg(feature = "with-runtime")]
    register_runtime_symbols(&engine);
    #[cfg(feature = "with-debug-utils")]
    metadata
        .get::<cairo_native::metadata::debug_utils::DebugUtils>()
        .unwrap()
        .register_impls(&engine);

    // Initialize arguments and return values.
    let params_input = match args.inputs {
        Some(StdioOrPath::Stdio) => Cow::Owned(io::read_to_string(io::stdin())?),
        Some(StdioOrPath::Path(path)) => Cow::Owned(fs::read_to_string(path)?),
        None => Cow::Borrowed("[]"),
    };
    let mut params = serde_json::Deserializer::new(StrRead::new(&params_input));

    match args.outputs {
        Some(StdioOrPath::Stdio) => {
            cairo_native::execute::<CoreType, CoreLibfunc, _, _>(
                &engine,
                &registry,
                &entry_point.id,
                &mut params,
                &mut serde_json::Serializer::pretty(io::stdout()),
                required_initial_gas,
            )
            .unwrap_or_else(|e| match &*e {
                cairo_native::error::jit_engine::ErrorImpl::DeserializeError(_) => {
                    panic!(
                        "Expected inputs with signature: ({})",
                        entry_point
                            .signature
                            .param_types
                            .iter()
                            .map(ToString::to_string)
                            .join(", ")
                    )
                }
                e => panic!("{:?}", e),
            });
            println!();
        }
        Some(StdioOrPath::Path(path)) => {
            let mut file = File::create(path)?;
            cairo_native::execute::<CoreType, CoreLibfunc, _, _>(
                &engine,
                &registry,
                &entry_point.id,
                &mut params,
                &mut serde_json::Serializer::pretty(&mut file),
                required_initial_gas,
            )
            .unwrap();
            writeln!(file)?;
        }
        None => {
            if args.print_outputs {
                todo!()
            }
        }
    }

    Ok(())
}

fn load_program(path: &Path) -> Result<Program, Box<dyn std::error::Error>> {
    Ok(match path.extension().and_then(OsStr::to_str) {
        Some("cairo") => {
            let mut db = RootDatabase::builder().detect_corelib().build()?;
            let main_crate_ids = setup_project(&mut db, path)?;
            (*compile_prepared_db(
                &mut db,
                main_crate_ids,
                CompilerConfig {
                    replace_ids: true,
                    ..Default::default()
                },
            )?)
            .clone()
        }
        Some("sierra") => {
            let program_src = fs::read_to_string(path)?;

            let program_parser = ProgramParser::new();
            program_parser
                .parse(&program_src)
                .map_err(|e| e.map_token(|t| t.to_string()))?
        }
        _ => unreachable!(),
    })
}

#[derive(Clone, Debug, Parser)]
struct CmdLine {
    #[clap(value_parser = parse_input)]
    input: PathBuf,

    #[clap(value_parser = parse_entry_point)]
    entry_point: FunctionId,

    #[clap(short = 'i', long = "inputs", value_parser = parse_io)]
    inputs: Option<StdioOrPath>,
    #[clap(short = 'o', long = "outputs", value_parser = parse_io)]
    outputs: Option<StdioOrPath>,
    #[clap(short = 'p', long = "print-outputs")]
    print_outputs: bool,
}

#[derive(Clone, Debug)]
enum StdioOrPath {
    Stdio,
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

fn parse_entry_point(input: &str) -> Result<FunctionId, Infallible> {
    Ok(match input.parse::<u64>() {
        Ok(id) => FunctionId::new(id),
        Err(_) => FunctionId::from_string(input),
    })
}

fn parse_io(input: &str) -> Result<StdioOrPath, String> {
    Ok(if input == "-" {
        StdioOrPath::Stdio
    } else {
        StdioOrPath::Path(match Path::new(input).extension().and_then(OsStr::to_str) {
            Some("json") => input.into(),
            _ => return Err("Input path expected to have `json` as its extension.".to_string()),
        })
    })
}
