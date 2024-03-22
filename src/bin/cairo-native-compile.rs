use anyhow::Context;
use cairo_lang_compiler::{compile_cairo_project_at_path, CompilerConfig};
use cairo_native::{
    context::NativeContext, metadata::MetadataStorage, module_to_object, object_to_shared_lib,
    OptLevel,
};
use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Clone, Debug, ValueEnum)]
enum RunMode {
    Aot,
    Jit,
}

/// Compiles a Cairo project outputting the generated MLIR and the shared library.
/// Exits with 1 if the compilation or run fails, otherwise 0.
#[derive(Parser, Debug)]
#[clap(version, verbatim_doc_comment)]
struct Args {
    /// The Cairo project path to compile and run its tests.
    path: PathBuf,
    /// Whether path is a single file.
    #[arg(short, long)]
    single_file: bool,
    /// Allows the compilation to succeed with warnings.
    #[arg(long)]
    allow_warnings: bool,
    /// Replaces sierra ids with human-readable ones.
    #[arg(short, long, default_value_t = false)]
    replace_ids: bool,
    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,
    /// The output path for the mlir, if none is passed, out.mlir will be the default.
    output_mlir: Option<PathBuf>,
    /// If a path is passed, a dynamic library will be compiled and saved at that path.
    output_library: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    let args = Args::parse();

    // Check if args.path is a file or a directory.
    check_compiler_path(args.single_file, &args.path)?;

    let sierra_program = compile_cairo_project_at_path(
        &args.path,
        CompilerConfig {
            replace_ids: args.replace_ids,
            ..CompilerConfig::default()
        },
    )?;
    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_module = native_context
        .compile(&sierra_program, MetadataStorage::default())
        .unwrap();

    let opt_level = match args.opt_level {
        0 => OptLevel::None,
        1 => OptLevel::Less,
        2 => OptLevel::Default,
        _ => OptLevel::Aggressive,
    };

    let output_mlir = args
        .output_mlir
        .unwrap_or_else(|| PathBuf::from("out.mlir"));

    std::fs::write(
        output_mlir,
        native_module.module().as_operation().to_string(),
    )
    .context("Failed to write output.")?;

    if let Some(output_library) = &args.output_library {
        let object_data = module_to_object(native_module.module(), opt_level)
            .context("Failed to convert module to object.")?;
        object_to_shared_lib(&object_data, output_library)
            .context("Failed to write shared library.")?;
    }

    Ok(())
}

pub fn check_compiler_path(single_file: bool, path: &Path) -> anyhow::Result<()> {
    if path.is_file() {
        if !single_file {
            anyhow::bail!("The given path is a file, but --single-file was not supplied.");
        }
    } else if path.is_dir() {
        if single_file {
            anyhow::bail!("The given path is a directory, but --single-file was supplied.");
        }
    } else {
        anyhow::bail!("The given path does not exist.");
    }
    Ok(())
}
