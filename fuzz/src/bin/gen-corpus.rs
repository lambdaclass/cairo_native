use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
};
use cairo_native_fuzz::{encode_value, is_builtin, is_supported, random_value};
use clap::Parser;
use rand::seq::SliceRandom;
use std::{error::Error, fs::File, io::Write, path::PathBuf};

/// Generate corpus for a Sierra program or contract class.
#[derive(Parser, Debug)]
enum Args {
    /// Generate corpus for Sierra program
    Program {
        /// Path to input Sierra program
        sierra_path: PathBuf,
        /// Path to corpus directory
        corpus_dir: PathBuf,
    },
    /// Generate corpus for Sierra contract class
    Contract {
        /// Path to input Sierra contract class
        contract_path: PathBuf,
        /// Path to corpus directory
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
        .collect::<Vec<_>>();
    funcs.shuffle(&mut rand::rng());

    for (idx, func, param_tys) in funcs
        .into_iter()
        .filter_map(|(idx, func)| {
            let mut param_tys = vec![];
            for param in &func.params {
                let param_ty = registry.get_type(&param.ty).unwrap();

                if is_builtin(param_ty) {
                    continue;
                } else if is_supported(param_ty, &registry) {
                    param_tys.push(param_ty)
                } else {
                    return None;
                };
            }

            Some((idx, func, param_tys))
        })
        .take(10)
    {
        let mut input_file = File::create(corpus_dir.join(format!("f{}", func.id.id))).unwrap();

        input_file.write_all(&(idx as u64).to_le_bytes()).unwrap();

        for param_ty in param_tys {
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
    todo!()
}
