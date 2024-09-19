use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
};
use cairo_lang_defs::plugin::NamedPlugin;
use cairo_lang_semantic::plugin::PluginSuite;
use cairo_lang_sierra::{program::Program, ProgramParser};
use cairo_lang_starknet::{
    compile::compile_contract_in_prepared_db, inline_macros::selector::SelectorMacro,
    plugin::StarkNetPlugin,
};
use cairo_native::context::NativeContext;
use clap::Parser;
use melior::ir::operation::OperationPrintingFlags;
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
    let context = NativeContext::new();
    let program = load_program(Path::new(&args.input), args.starknet)?;

    // Compile the program.
    let module = context.compile(&program, false)?;

    // Write the output.
    let output_str = module
        .module()
        .as_operation()
        .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))?;
    match args.output {
        CompilerOutput::Stdout => println!("{output_str}"),
        CompilerOutput::Path(path) => fs::write(path, &output_str)?,
    }

    Ok(())
}

fn load_program(path: &Path, is_contract: bool) -> Result<Program, Box<dyn std::error::Error>> {
    Ok(match path.extension().and_then(OsStr::to_str) {
        Some("cairo") if !is_contract => {
            let mut db = RootDatabase::builder().detect_corelib().build()?;
            let main_crate_ids = setup_project(&mut db, path)?;

            compile_prepared_db(
                &db,
                main_crate_ids,
                CompilerConfig {
                    replace_ids: true,
                    ..Default::default()
                },
            )?
            .program
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

            contract.extract_sierra_program()?
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
