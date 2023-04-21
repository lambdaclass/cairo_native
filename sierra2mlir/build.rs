use std::{
    env, fs,
    process::{Command, Stdio},
};

fn main() {
    let profile = env::var("PROFILE").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();

    assert!(Command::new("cargo")
        .arg("build")
        .arg(format!("--target-dir={out_dir}"))
        .arg(&format!(
            "--profile={}",
            match profile.as_str() {
                "debug" => "dev",
                x => x,
            }
        ))
        .arg("-p=sierra2mlir-utils")
        .spawn()
        .unwrap()
        .wait()
        .unwrap()
        .success());

    println!("cargo:rustc-env=S2M_UTILS_PATH={out_dir}/{profile}/libsierra2mlir_utils.so");
    println!(
        "cargo:rustc-env=SHARED_LIB_EXT={}",
        match env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
            "linux" => "so",
            "macos" => "dylib",
            "windows" => "dll",
            _ => panic!("Unsupported OS."),
        }
    );
}
