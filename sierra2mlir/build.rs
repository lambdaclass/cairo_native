use std::{env, process::Command};

fn main() {
    // Hack to make rust-analyzer load the project.
    if env::var("RUSTC_WRAPPER").unwrap().ends_with("rust-analyzer") {
        return;
    }

    let target_dir = env::var("CARGO_TARGET_DIR").unwrap();
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

    println!(
        "cargo:rustc-env=S2M_UTILS_PATH=target/{target_dir}/{profile}/libsierra2mlir_utils.so"
    );
}
