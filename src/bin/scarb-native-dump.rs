mod utils;

use std::{env, fs};

use anyhow::Context;
use cairo_lang_sierra::program::VersionedProgram;
use cairo_native::context::NativeContext;
use melior::ir::operation::OperationPrintingFlags;
use scarb_metadata::{MetadataCommand, ScarbCommand};

/// Compiles all packages from a Scarb project on the current directory.

fn main() -> anyhow::Result<()> {
    let metadata = MetadataCommand::new().inherit_stderr().exec()?;

    // Build only the filtered packages.
    ScarbCommand::new().arg("build").run()?;

    // Get `target` directory.
    let profile = env::var("SCARB_PROFILE").unwrap_or("dev".into());
    let default_target_dir = metadata.runtime_manifest.join("target");
    let target_dir = metadata
        .target_dir
        .clone()
        .unwrap_or(default_target_dir)
        .join(profile);

    let native_context = NativeContext::new();
    for package in metadata.packages.iter() {
        for target in &package.targets {
            let file_path = target_dir.join(format!("{}.sierra.json", target.name.clone()));

            if file_path.exists() {
                let compiled = serde_json::from_str::<VersionedProgram>(
                    &fs::read_to_string(file_path.clone())
                        .with_context(|| format!("failed to read file: {file_path}"))?,
                )
                .with_context(|| format!("failed to deserialize compiled file: {file_path}"))?;

                // Compile the sierra program into a MLIR module.
                let native_module = native_context
                    .compile(&compiled.into_v1().unwrap().program, None)
                    .unwrap();

                // Write the output.
                let output_str = native_module.module().as_operation().to_string_with_flags(
                    OperationPrintingFlags::new().enable_debug_info(true, false),
                )?;

                fs::write(
                    target_dir.join(format!("{}.mlir", target.name.clone())),
                    &output_str,
                )?;
            }
        }
    }

    Ok(())
}
// PLT: ACK
