//use anyhow::Context;
use anyhow::Context;
//use cairo_native::{
use cairo_native::{
//    context::NativeContext, module_to_object, object_to_shared_lib,
    context::NativeContext, module_to_object, object_to_shared_lib,
//    utils::cairo_to_sierra_with_debug_info,
    utils::cairo_to_sierra_with_debug_info,
//};
};
//use clap::{Parser, ValueEnum};
use clap::{Parser, ValueEnum};
//

//use std::path::{Path, PathBuf};
use std::path::{Path, PathBuf};
//use tracing_subscriber::{EnvFilter, FmtSubscriber};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
//

//#[derive(Clone, Debug, ValueEnum)]
#[derive(Clone, Debug, ValueEnum)]
//enum RunMode {
enum RunMode {
//    Aot,
    Aot,
//    Jit,
    Jit,
//}
}
//

///// Compiles a Cairo project outputting the generated MLIR and the shared library.
/// Compiles a Cairo project outputting the generated MLIR and the shared library.
///// Exits with 1 if the compilation or run fails, otherwise 0.
/// Exits with 1 if the compilation or run fails, otherwise 0.
//#[derive(Parser, Debug)]
#[derive(Parser, Debug)]
//#[clap(version, verbatim_doc_comment)]
#[clap(version, verbatim_doc_comment)]
//struct Args {
struct Args {
//    /// The Cairo project path to compile and run its tests.
    /// The Cairo project path to compile and run its tests.
//    path: PathBuf,
    path: PathBuf,
//    /// Whether path is a single file.
    /// Whether path is a single file.
//    #[arg(short, long)]
    #[arg(short, long)]
//    single_file: bool,
    single_file: bool,
//    /// Allows the compilation to succeed with warnings.
    /// Allows the compilation to succeed with warnings.
//    #[arg(long)]
    #[arg(long)]
//    allow_warnings: bool,
    allow_warnings: bool,
//    /// Replaces sierra ids with human-readable ones.
    /// Replaces sierra ids with human-readable ones.
//    #[arg(short, long, default_value_t = false)]
    #[arg(short, long, default_value_t = false)]
//    replace_ids: bool,
    replace_ids: bool,
//    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
//    #[arg(short = 'O', long, default_value_t = 0)]
    #[arg(short = 'O', long, default_value_t = 0)]
//    opt_level: u8,
    opt_level: u8,
//    /// The output path for the mlir, if none is passed, out.mlir will be the default.
    /// The output path for the mlir, if none is passed, out.mlir will be the default.
//    output_mlir: Option<PathBuf>,
    output_mlir: Option<PathBuf>,
//    /// If a path is passed, a dynamic library will be compiled and saved at that path.
    /// If a path is passed, a dynamic library will be compiled and saved at that path.
//    output_library: Option<PathBuf>,
    output_library: Option<PathBuf>,
//}
}
//

//fn main() -> anyhow::Result<()> {
fn main() -> anyhow::Result<()> {
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

//    let args = Args::parse();
    let args = Args::parse();
//

//    // Check if args.path is a file or a directory.
    // Check if args.path is a file or a directory.
//    check_compiler_path(args.single_file, &args.path)?;
    check_compiler_path(args.single_file, &args.path)?;
//

//    let native_context = NativeContext::new();
    let native_context = NativeContext::new();
//    let (sierra_program, debug_locations) =
    let (sierra_program, debug_locations) =
//        cairo_to_sierra_with_debug_info(native_context.context(), &args.path)?;
        cairo_to_sierra_with_debug_info(native_context.context(), &args.path)?;
//

//    // Compile the sierra program into a MLIR module.
    // Compile the sierra program into a MLIR module.
//    let native_module = native_context
    let native_module = native_context
//        .compile(&sierra_program, Some(debug_locations))
        .compile(&sierra_program, Some(debug_locations))
//        .unwrap();
        .unwrap();
//

//    let output_mlir = args
    let output_mlir = args
//        .output_mlir
        .output_mlir
//        .unwrap_or_else(|| PathBuf::from("out.mlir"));
        .unwrap_or_else(|| PathBuf::from("out.mlir"));
//

//    std::fs::write(
    std::fs::write(
//        output_mlir,
        output_mlir,
//        native_module.module().as_operation().to_string(),
        native_module.module().as_operation().to_string(),
//    )
    )
//    .context("Failed to write output.")?;
    .context("Failed to write output.")?;
//

//    if let Some(output_library) = &args.output_library {
    if let Some(output_library) = &args.output_library {
//        let object_data = module_to_object(native_module.module(), args.opt_level.into())
        let object_data = module_to_object(native_module.module(), args.opt_level.into())
//            .context("Failed to convert module to object.")?;
            .context("Failed to convert module to object.")?;
//        object_to_shared_lib(&object_data, output_library)
        object_to_shared_lib(&object_data, output_library)
//            .context("Failed to write shared library.")?;
            .context("Failed to write shared library.")?;
//    }
    }
//

//    Ok(())
    Ok(())
//}
}
//

//pub fn check_compiler_path(single_file: bool, path: &Path) -> anyhow::Result<()> {
pub fn check_compiler_path(single_file: bool, path: &Path) -> anyhow::Result<()> {
//    if path.is_file() {
    if path.is_file() {
//        if !single_file {
        if !single_file {
//            anyhow::bail!("The given path is a file, but --single-file was not supplied.");
            anyhow::bail!("The given path is a file, but --single-file was not supplied.");
//        }
        }
//    } else if path.is_dir() {
    } else if path.is_dir() {
//        if single_file {
        if single_file {
//            anyhow::bail!("The given path is a directory, but --single-file was supplied.");
            anyhow::bail!("The given path is a directory, but --single-file was supplied.");
//        }
        }
//    } else {
    } else {
//        anyhow::bail!("The given path does not exist.");
        anyhow::bail!("The given path does not exist.");
//    }
    }
//    Ok(())
    Ok(())
//}
}
