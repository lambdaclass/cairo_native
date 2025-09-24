use anyhow::Context;
use cairo_lang_compiler::project::check_compiler_path;
use cairo_native::{
    clone_option_mut, context::NativeContext, module_to_object, object_to_shared_lib,
    statistics::Statistics, utils::testing::cairo_to_sierra,
};
use clap::Parser;
use std::{
    fs::{self, File},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

/// Compiles Cairo/Sierra to Native machine code.
/// Outputs the generated MLIR, and the final shared library.
///
/// Supports different types of inputs:
/// - Cairo project (default)
/// - Standalone Cairo file (with the --single-file option).
/// - Standalone Sierra JSON file (with the --sierra-json option).
///
/// There is no easy way of executing the compiled shared library, so this
/// binary is mostly used for debugging compilation.
///
/// Exits with 1 if the compilation or run fails, otherwise 0.
#[derive(Parser, Debug)]
#[clap(version, verbatim_doc_comment)]
struct Args {
    /// The input path to compile.
    /// By default, it is intrepreted as a Cairo project.
    path: PathBuf,
    /// Whether path is a single Cairo file.
    #[arg(short, long, group = "input")]
    single_file: bool,
    /// Whether path is a single Sierra JSON file.
    #[arg(long, group = "input")]
    sierra_json: bool,
    /// Optimization level (Valid: 0, 1, 2, 3).
    /// Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,
    /// The output path for the generated MLIR.
    #[arg(default_value = "out.mlir")]
    output_mlir: PathBuf,
    /// The output path for the shared library.
    #[cfg_attr(target_os = "macos", arg(default_value = "out.dylib"))]
    #[cfg_attr(not(target_os = "macos"), arg(default_value = "out.so"))]
    output_library: PathBuf,
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

    // First, we obtain the sierra program. If the --sierra-json file was
    // given, we read the input as Sierra directly. Otherwise, we compile Cairo
    // to Sierra.
    let sierra_program = if args.sierra_json {
        let sierra_file = File::open(&args.path)?;
        let program =
            serde_json::from_reader(sierra_file).context("Failed to read Sierra JSON file.")?;
        Arc::new(program)
    } else {
        // Check if args.path is a file or a directory.
        check_compiler_path(args.single_file, &args.path)?;
        cairo_to_sierra(&args.path).context("Failed to compile Cairo to Sierra.")?
    };

    let mut stats_with_path = args.stats_path.map(|path| (Statistics::default(), path));
    let stats = stats_with_path.as_mut().map(|v| &mut v.0);

    let pre_compilation_instant = Instant::now();

    // Compile the sierra program into a MLIR module.
    let native_context = NativeContext::new();
    let native_module = native_context
        .compile(
            &sierra_program,
            false,
            Some(Default::default()),
            clone_option_mut!(stats),
        )
        .context("Failed to compile to MLIR.")?;
    std::fs::write(
        args.output_mlir,
        native_module.module().as_operation().to_string(),
    )
    .context("Failed to write MLIR output.")?;

    let object_data = module_to_object(
        native_module.module(),
        args.opt_level.into(),
        clone_option_mut!(stats),
    )
    .context("Failed to convert MLIR to object.")?;
    object_to_shared_lib(&object_data, &args.output_library, clone_option_mut!(stats))
        .context("Failed to write shared library.")?;

    let compilation_time = pre_compilation_instant.elapsed().as_millis();
    if let Some(&mut ref mut stats) = stats {
        stats.compilation_total_time_ms = Some(compilation_time);
    }

    if let Some((stats, stats_path)) = stats_with_path {
        fs::write(stats_path, serde_json::to_string_pretty(&stats)?)?;
    }

    Ok(())
}
