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
use dlopen2::raw::Library;
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
    mem::ManuallyDrop,
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input .sierra or .cairo program to know the types to use in the shared library.
    input: PathBuf,

    /// Input compiled library
    lib: PathBuf,

    sym: String,

    /// Whether the program is a contract.
    #[arg(short, long)]
    starknet: bool,
}

/*
typedef struct factorial_return_values
{
    unsigned __int128 remaining_gas;
    struct {
        uint8_t discriminant;
        union {
            uint64_t ok[4];
            struct {
                void* ptr;
                uint32_t len;
                uint32_t cap;
            } err;
        };
    } result;
} factorial_return_values_t;
 */

#[derive(Debug)]
#[repr(C)]
struct ResultError {
    ptr: *const (),
    len: u32,
    cap: u32,
}

#[repr(C)]
union ResultUnion {
    ok: [u64; 4],
    error: ManuallyDrop<ResultError>,
}

#[repr(C)]
struct ResultEnum {
    discriminant: u8,
    reuslt: ResultUnion,
}

#[repr(C)]
struct ReturnValues {
    remaining_gas: u128,
    result: ResultEnum,
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

    let lib = Library::open(args.input)?;
    let sym: extern "C" fn() -> ReturnValues = unsafe { lib.symbol(&args.sym).unwrap() };

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
