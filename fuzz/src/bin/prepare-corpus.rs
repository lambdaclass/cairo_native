use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    program::Program,
    program_registry::ProgramRegistry,
};
use cairo_native_fuzz::{encode_value, is_builtin, is_supported, random_value};
use clap::{command, Parser};
use rand::seq::SliceRandom;
use std::{fs::File, io::Write, path::PathBuf};

/// Generate an example Corpus for running the given Sierra program.
#[derive(Parser, Debug)]
#[command()]
struct Args {
    /// Path to input Sierra program
    sierra_path: PathBuf,
    /// Path to corpus directory
    corpus_dir: PathBuf,
}

fn main() {
    let args = Args::parse();

    let program_file = File::open(args.sierra_path).unwrap();
    let program: Program = serde_json::from_reader(program_file).unwrap();
    let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

    std::fs::create_dir_all(&args.corpus_dir).unwrap();

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
        let mut input_file =
            File::create(args.corpus_dir.join(format!("f{}", func.id.id))).unwrap();

        input_file.write_all(&(idx as u64).to_le_bytes()).unwrap();

        for param_ty in param_tys {
            let value = random_value(param_ty, &registry);
            encode_value(&value, &mut input_file).unwrap();
        }
    }
}
