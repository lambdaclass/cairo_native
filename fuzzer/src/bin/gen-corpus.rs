use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
};
use cairo_lang_starknet_classes::contract_class::ContractClass;
use cairo_native_fuzzer::{encode_value, is_function_supported, random_value};
use clap::Parser;
use rand::seq::SliceRandom;
use starknet_types_core::felt::Felt;
use std::{error::Error, fs::File, io::Write, path::PathBuf};

/// Generate corpus for the fuzzer.
///
/// Randomly selects entrypoints and generates fuzz
/// inputs for executing those entrypoints.
#[derive(Parser, Debug)]
enum Args {
    /// Generate corpus for Sierra program.
    Program {
        /// Path to input Sierra program.
        sierra_path: PathBuf,
        /// Path to corpus directory.
        corpus_dir: PathBuf,
    },
    /// Generate corpus for Sierra contract class.
    Contract {
        /// Path to input Sierra contract class.
        contract_path: PathBuf,
        /// Path to corpus directory.
        corpus_dir: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    match args {
        Args::Program {
            sierra_path,
            corpus_dir,
        } => generate_program_corpus(sierra_path, corpus_dir)?,
        Args::Contract {
            contract_path,
            corpus_dir,
        } => generate_contract_corpus(contract_path, corpus_dir)?,
    }

    Ok(())
}

fn generate_program_corpus(
    sierra_path: PathBuf,
    corpus_dir: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let program_file = File::open(sierra_path).unwrap();
    let program: Program = serde_json::from_reader(program_file).unwrap();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

    std::fs::create_dir_all(&corpus_dir).unwrap();

    let mut funcs = program
        .funcs
        .iter()
        .cloned()
        .enumerate()
        .filter(|(_, func)| is_function_supported(func, &registry))
        .collect::<Vec<_>>();
    funcs.shuffle(&mut rand::rng());

    for (idx, func) in funcs.into_iter().take(10) {
        let mut input_file = File::create(corpus_dir.join(format!("f{}", func.id.id))).unwrap();

        input_file.write_all(&(idx as u64).to_le_bytes()).unwrap();

        for param_ty_id in &func.signature.param_types {
            let param_ty = registry.get_type(param_ty_id).unwrap();
            let value = random_value(param_ty, &registry);
            encode_value(&value, &mut input_file).unwrap();
        }
    }

    Ok(())
}

fn generate_contract_corpus(
    contract_path: PathBuf,
    corpus_dir: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let contract_file = File::open(contract_path).unwrap();
    let contract: ContractClass = serde_json::from_reader(contract_file).unwrap();

    std::fs::create_dir_all(&corpus_dir).unwrap();

    let mut entrypoints = []
        .iter()
        .chain(&contract.entry_points_by_type.external)
        .chain(&contract.entry_points_by_type.l1_handler)
        .chain(&contract.entry_points_by_type.constructor)
        .collect::<Vec<_>>();
    entrypoints.shuffle(&mut rand::rng());

    for entrypoint in entrypoints.into_iter().take(10) {
        let mut input_file =
            File::create(corpus_dir.join(format!("f0x{:x}", entrypoint.selector))).unwrap();

        let selector_felt = Felt::from(entrypoint.selector.clone());
        input_file.write_all(&selector_felt.to_bytes_le())?;

        let length = rand::random::<u8>() % 32;
        input_file.write_all(&length.to_le_bytes())?;
        for _ in 0..length {
            input_file.write_all(&Felt::from_bytes_le(&rand::random()).to_bytes_le())?;
        }
    }

    Ok(())
}
