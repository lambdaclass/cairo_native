use anyhow::{anyhow, bail, Context};
use std::path::PathBuf;

use cairo_lang_sierra::program::Program;
use cairo_lang_starknet_classes::compiler_version::VersionId;
use cairo_lang_starknet_classes::contract_class::ContractClass;
use cairo_native::executor::AotContractExecutor;
use cairo_native::OptLevel;
use clap::Parser;

/// Given a Sierra file (as saved in Starknet's contract tree), extracts the sierra_program from
/// felts into readable Sierra code, compiles it to native, and saves the result to the given output
/// path.
#[derive(Parser, Debug)]
#[clap(version, verbatim_doc_comment)]
struct Args {
    /// The path of the Sierra file to compile.
    path: PathBuf,
    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,
    /// The output file path.
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let (contract_class, sierra_program, sierra_version) =
        load_sierra_program_from_file(&args.path)?;

    AotContractExecutor::new_into(
        &sierra_program,
        &contract_class.entry_points_by_type,
        sierra_version,
        args.output.clone(),
        args.opt_level.into(),
    )
    .context("Error compiling Sierra program.")?
    .with_context(|| format!("Failed to take lock on path {}", args.output.display()))?;
    Ok(())
}

/// Extracts the first 3 felts from the Sierra program and parses them into a VersionId.
fn get_sierra_version_from_program<F>(sierra_program: &[F]) -> anyhow::Result<VersionId>
where
    F: TryInto<usize> + std::fmt::Display + Clone,
    <F as TryInto<usize>>::Error: std::fmt::Display,
{
    if sierra_program.len() < 3 {
        bail!("Sierra program length must be at least 3 Felts.");
    }

    let version_components: Vec<usize> = sierra_program
        .iter()
        .take(3)
        .enumerate()
        .map(|(index, felt)| {
            felt.clone().try_into().map_err(|err| {
                anyhow!(
                    "Failed to parse Sierra program to Sierra version. Index: {}, Felt: {}, \
                        Error: {}",
                    index,
                    felt,
                    err
                )
            })
        })
        .collect::<Result<_, _>>()?;

    Ok(VersionId {
        major: version_components[0],
        minor: version_components[1],
        patch: version_components[2],
    })
}

/// Given a Sierra file path, loads the contract class from it, extracts the sierra version from the
/// first 3 felts of the compressed sierra_program, and extracts the compressed sierra_program into
/// readable sierra code.
fn load_sierra_program_from_file(
    path: &PathBuf,
) -> anyhow::Result<(ContractClass, Program, VersionId)> {
    let raw_contract_class = std::fs::read_to_string(path).context("Error reading Sierra file.")?;

    let contract_class: ContractClass = serde_json::from_str(&raw_contract_class)
        .context("Error deserializing Sierra file into contract class.")?;
    let raw_sierra_program: Vec<_> = contract_class
        .sierra_program
        .iter()
        .map(|big_uint_as_hex| big_uint_as_hex.value.clone())
        .collect();

    let sierra_version = get_sierra_version_from_program(&raw_sierra_program)?;
    Ok((
        contract_class.clone(),
        contract_class
            .extract_sierra_program()
            .context("Error extracting Sierra program from contract class.")?,
        sierra_version,
    ))
}
