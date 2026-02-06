use cairo_lang_compiler::CompilerConfig;
use cairo_lang_starknet::compile::compile_path;
use clap::Parser;
use std::{error::Error, fs::File, path::PathBuf};

/// Compiles Cairo contract to Sierra
#[derive(Parser, Debug)]
#[command()]
struct Args {
    /// Path to input Cairo program
    cairo_path: PathBuf,
    /// Path to output Sierra contract class
    sierra_contract_path: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let contract_class = compile_path(
        &args.cairo_path,
        None,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
        Default::default(),
    )?;

    let sierra_contract_file = File::create(&args.sierra_contract_path)?;
    serde_json::to_writer_pretty(sierra_contract_file, &contract_class)?;

    Ok(())
}
