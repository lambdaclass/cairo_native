//use cairo_lang_compiler::{
use cairo_lang_compiler::{
//    compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
    compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
//    project::setup_project, CompilerConfig,
    project::setup_project, CompilerConfig,
//};
};
//use cairo_lang_defs::plugin::NamedPlugin;
use cairo_lang_defs::plugin::NamedPlugin;
//use cairo_lang_semantic::plugin::PluginSuite;
use cairo_lang_semantic::plugin::PluginSuite;
//use cairo_lang_sierra::{program::Program, ProgramParser};
use cairo_lang_sierra::{program::Program, ProgramParser};
//use cairo_lang_starknet::{
use cairo_lang_starknet::{
//    compile::compile_contract_in_prepared_db, inline_macros::selector::SelectorMacro,
    compile::compile_contract_in_prepared_db, inline_macros::selector::SelectorMacro,
//    plugin::StarkNetPlugin,
    plugin::StarkNetPlugin,
//};
};
//use cairo_native::{
use cairo_native::{
//    context::NativeContext,
    context::NativeContext,
//    debug_info::{DebugInfo, DebugLocations},
    debug_info::{DebugInfo, DebugLocations},
//};
};
//use clap::Parser;
use clap::Parser;
//use melior::{ir::operation::OperationPrintingFlags, Context};
use melior::{ir::operation::OperationPrintingFlags, Context};
//use std::{
use std::{
//    ffi::OsStr,
    ffi::OsStr,
//    fs,
    fs,
//    path::{Path, PathBuf},
    path::{Path, PathBuf},
//    sync::Arc,
    sync::Arc,
//};
};
//use tracing_subscriber::{EnvFilter, FmtSubscriber};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
//

//fn main() -> Result<(), Box<dyn std::error::Error>> {
fn main() -> Result<(), Box<dyn std::error::Error>> {
//    // Parse command-line arguments.
    // Parse command-line arguments.
//    let args = CmdLine::parse();
    let args = CmdLine::parse();
//

//    // Configure logging and error handling.
    // Configure logging and error handling.
//    tracing::subscriber::set_global_default(
    tracing::subscriber::set_global_default(
//        FmtSubscriber::builder()
        FmtSubscriber::builder()
//            .with_env_filter(EnvFilter::from_default_env())
            .with_env_filter(EnvFilter::from_default_env())
//            .finish(),
            .finish(),
//    )?;
    )?;
//

//    // Load the program.
    // Load the program.
//    let context = NativeContext::new();
    let context = NativeContext::new();
//    // TODO: Reconnect debug information.
    // TODO: Reconnect debug information.
//    let (program, debug_info) = load_program(
    let (program, debug_info) = load_program(
//        Path::new(&args.input),
        Path::new(&args.input),
//        Some(context.context()),
        Some(context.context()),
//        args.starknet,
        args.starknet,
//    )?;
    )?;
//

//    // Compile the program.
    // Compile the program.
//    let module = context.compile(&program, debug_info)?;
    let module = context.compile(&program, debug_info)?;
//

//    // Write the output.
    // Write the output.
//    let output_str = module
    let output_str = module
//        .module()
        .module()
//        .as_operation()
        .as_operation()
//        .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))?;
        .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))?;
//    match args.output {
    match args.output {
//        CompilerOutput::Stdout => println!("{output_str}"),
        CompilerOutput::Stdout => println!("{output_str}"),
//        CompilerOutput::Path(path) => fs::write(path, &output_str)?,
        CompilerOutput::Path(path) => fs::write(path, &output_str)?,
//    }
    }
//

//    Ok(())
    Ok(())
//}
}
//

//fn load_program<'c>(
fn load_program<'c>(
//    path: &Path,
    path: &Path,
//    context: Option<&'c Context>,
    context: Option<&'c Context>,
//    is_contract: bool,
    is_contract: bool,
//) -> Result<(Program, Option<DebugLocations<'c>>), Box<dyn std::error::Error>> {
) -> Result<(Program, Option<DebugLocations<'c>>), Box<dyn std::error::Error>> {
//    Ok(match path.extension().and_then(OsStr::to_str) {
    Ok(match path.extension().and_then(OsStr::to_str) {
//        Some("cairo") if !is_contract => {
        Some("cairo") if !is_contract => {
//            let mut db = RootDatabase::builder().detect_corelib().build()?;
            let mut db = RootDatabase::builder().detect_corelib().build()?;
//            let main_crate_ids = setup_project(&mut db, path)?;
            let main_crate_ids = setup_project(&mut db, path)?;
//            let program = compile_prepared_db(
            let program = compile_prepared_db(
//                &mut db,
                &mut db,
//                main_crate_ids,
                main_crate_ids,
//                CompilerConfig {
                CompilerConfig {
//                    replace_ids: true,
                    replace_ids: true,
//                    ..Default::default()
                    ..Default::default()
//                },
                },
//            )?;
            )?;
//

//            let debug_locations = if let Some(context) = context {
            let debug_locations = if let Some(context) = context {
//                let debug_info = DebugInfo::extract(&db, &program).map_err(|_| {
                let debug_info = DebugInfo::extract(&db, &program).map_err(|_| {
//                    let mut buffer = String::new();
                    let mut buffer = String::new();
//                    assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
                    assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
//                    buffer
                    buffer
//                })?;
                })?;
//

//                Some(DebugLocations::extract(context, &db, &debug_info))
                Some(DebugLocations::extract(context, &db, &debug_info))
//            } else {
            } else {
//                None
                None
//            };
            };
//

//            (program, debug_locations)
            (program, debug_locations)
//        }
        }
//        Some("cairo") if is_contract => {
        Some("cairo") if is_contract => {
//            // mimics cairo_lang_starknet::contract_class::compile_path
            // mimics cairo_lang_starknet::contract_class::compile_path
//            let mut plugins = PluginSuite::default();
            let mut plugins = PluginSuite::default();
//            plugins
            plugins
//                .add_plugin::<StarkNetPlugin>()
                .add_plugin::<StarkNetPlugin>()
//                .add_inline_macro_plugin_ex(SelectorMacro::NAME, Arc::new(SelectorMacro));
                .add_inline_macro_plugin_ex(SelectorMacro::NAME, Arc::new(SelectorMacro));
//            let mut db = RootDatabase::builder()
            let mut db = RootDatabase::builder()
//                .detect_corelib()
                .detect_corelib()
//                .with_plugin_suite(plugins)
                .with_plugin_suite(plugins)
//                .build()?;
                .build()?;
//

//            let main_crate_ids = setup_project(&mut db, Path::new(&path))?;
            let main_crate_ids = setup_project(&mut db, Path::new(&path))?;
//

//            let contract = compile_contract_in_prepared_db(
            let contract = compile_contract_in_prepared_db(
//                &db,
                &db,
//                None,
                None,
//                main_crate_ids,
                main_crate_ids,
//                CompilerConfig {
                CompilerConfig {
//                    replace_ids: true,
                    replace_ids: true,
//                    ..Default::default()
                    ..Default::default()
//                },
                },
//            )?;
            )?;
//

//            let program = contract.extract_sierra_program()?;
            let program = contract.extract_sierra_program()?;
//

//            let debug_locations = if let Some(context) = context {
            let debug_locations = if let Some(context) = context {
//                let debug_info = DebugInfo::extract(&db, &program).map_err(|_| {
                let debug_info = DebugInfo::extract(&db, &program).map_err(|_| {
//                    let mut buffer = String::new();
                    let mut buffer = String::new();
//                    assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
                    assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
//                    buffer
                    buffer
//                })?;
                })?;
//

//                Some(DebugLocations::extract(context, &db, &debug_info))
                Some(DebugLocations::extract(context, &db, &debug_info))
//            } else {
            } else {
//                None
                None
//            };
            };
//

//            (program, debug_locations)
            (program, debug_locations)
//        }
        }
//        Some("sierra") => {
        Some("sierra") => {
//            let program_src = fs::read_to_string(path)?;
            let program_src = fs::read_to_string(path)?;
//

//            let program_parser = ProgramParser::new();
            let program_parser = ProgramParser::new();
//            let program = program_parser
            let program = program_parser
//                .parse(&program_src)
                .parse(&program_src)
//                .map_err(|e| e.map_token(|t| t.to_string()))?;
                .map_err(|e| e.map_token(|t| t.to_string()))?;
//

//            (program, None)
            (program, None)
//        }
        }
//        _ => unreachable!(),
        _ => unreachable!(),
//    })
    })
//}
}
//

//#[derive(Clone, Debug, Parser)]
#[derive(Clone, Debug, Parser)]
//struct CmdLine {
struct CmdLine {
//    #[clap(value_parser = parse_input)]
    #[clap(value_parser = parse_input)]
//    input: PathBuf,
    input: PathBuf,
//

//    #[clap(short = 'o', long = "output", value_parser = parse_output, default_value = "-")]
    #[clap(short = 'o', long = "output", value_parser = parse_output, default_value = "-")]
//    output: CompilerOutput,
    output: CompilerOutput,
//

//    /// Compile a starknet contract
    /// Compile a starknet contract
//    #[clap(long)]
    #[clap(long)]
//    starknet: bool,
    starknet: bool,
//}
}
//

//#[derive(Clone, Debug)]
#[derive(Clone, Debug)]
//enum CompilerOutput {
enum CompilerOutput {
//    Stdout,
    Stdout,
//    Path(PathBuf),
    Path(PathBuf),
//}
}
//

//fn parse_input(input: &str) -> Result<PathBuf, String> {
fn parse_input(input: &str) -> Result<PathBuf, String> {
//    Ok(match Path::new(input).extension().and_then(OsStr::to_str) {
    Ok(match Path::new(input).extension().and_then(OsStr::to_str) {
//        Some("cairo" | "sierra") => input.into(),
        Some("cairo" | "sierra") => input.into(),
//        _ => {
        _ => {
//            return Err(
            return Err(
//                "Input path expected to have either `cairo` or `sierra` as its extension."
                "Input path expected to have either `cairo` or `sierra` as its extension."
//                    .to_string(),
                    .to_string(),
//            )
            )
//        }
        }
//    })
    })
//}
}
//

//fn parse_output(input: &str) -> Result<CompilerOutput, String> {
fn parse_output(input: &str) -> Result<CompilerOutput, String> {
//    Ok(if input == "-" {
    Ok(if input == "-" {
//        CompilerOutput::Stdout
        CompilerOutput::Stdout
//    } else {
    } else {
//        CompilerOutput::Path(match Path::new(input).extension().and_then(OsStr::to_str) {
        CompilerOutput::Path(match Path::new(input).extension().and_then(OsStr::to_str) {
//            Some("mlir") => input.into(),
            Some("mlir") => input.into(),
//            _ => {
            _ => {
//                return Err(
                return Err(
//                    "Output path expected to be `-` for stdout or have `mlir` as its extension."
                    "Output path expected to be `-` for stdout or have `mlir` as its extension."
//                        .to_string(),
                        .to_string(),
//                )
                )
//            }
            }
//        })
        })
//    })
    })
//}
}
