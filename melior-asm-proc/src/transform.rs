use proc_macro2::TokenStream;
use quote::spanned::Spanned;
use std::{
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

// TODO: Check if the error diagnostic spans match the source or should be updated.
pub fn transform_with_opt(source: TokenStream, flags: &[impl AsRef<str>]) -> String {
    let mlir_opt_path = match std::env::var_os("MLIR_SYS_160_PREFIX") {
        Some(x) => Path::new(x.to_str().unwrap()).join("bin/mlir-opt"),
        None => Path::new("mlir-opt").to_path_buf(),
    };

    let mut child = Command::new(mlir_opt_path)
        .args(&flags.iter().map(|x| x.as_ref()).collect::<Vec<_>>())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.__span().source_text().unwrap().as_bytes())
        .unwrap();

    let child_result = child.wait_with_output().unwrap();
    if !child_result.status.success() {
        todo!()
    }

    String::from_utf8(child_result.stdout).unwrap()
}
