use cairo_lang_sierra::program::Program;
use cairo_native::{
    clone_option_mut, context::NativeContext, module_to_object, object_to_shared_lib,
    statistics::Statistics,
};
use clap::Parser;
use std::{
    fs::{self, File},
    path::PathBuf,
};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

/// This binary compiles sierra directly, instead of receiving the Cairo file/project.
/// It is used for debugging the compilation of programs, and not for executing them.
///
/// It can be used with Scarb to compile Cairo projects. Scarb generates
/// Sierra json files, which can then be compiled with this tool.
///
/// # Example
///
/// To compile a scarb project with Native, run:
///
/// ```sh
/// scarb build
/// cairo-native-sierra-compile target/dev/a.sierra.json a.so
/// ```
///
/// NOTE: The `a.sierra.json` won't be generated for starknet contracts.
#[derive(Parser, Debug)]
#[clap(version, verbatim_doc_comment)]
struct Args {
    /// The sierra json path to compile.
    sierra_path: PathBuf,
    /// The compiled output path.
    output: PathBuf,
    /// Optimization level (Valid: 0, 1, 2, 3).
    /// Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,
    /// The compilation statistics path.
    #[arg(long)]
    stats_path: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    let args = Args::parse();

    // Read the sierra program.
    let sierra_file = File::open(args.sierra_path)?;
    let sierra_program: Program = serde_json::from_reader(sierra_file)?;

    let mut stats_with_path = args.stats_path.map(|path| (Statistics::default(), path));
    let stats = stats_with_path.as_mut().map(|v| &mut v.0);

    // Compile the sierra program into a MLIR module.
    let native_context = NativeContext::new();
    let native_module = native_context.compile(
        &sierra_program,
        false,
        Some(Default::default()),
        clone_option_mut!(stats),
    )?;

    // Compile the MLIR module to object.
    let object_data = module_to_object(
        native_module.module(),
        args.opt_level.into(),
        clone_option_mut!(stats),
    )?;

    // Link the object as a shared library.
    object_to_shared_lib(&object_data, &args.output, stats)?;

    if let Some((stats, stats_path)) = stats_with_path {
        fs::write(stats_path, serde_json::to_string_pretty(&stats)?)?;
    }

    Ok(())
}
