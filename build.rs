use std::env::var;

fn main() {
    cc::Build::new()
        .cpp(true)
        .flag_if_supported("-std=c++17")
        .flag_if_supported(&format!("-I{}/include", var("MLIR_SYS_160_PREFIX").unwrap()))
        .flag_if_supported("-Wno-unused-parameter")
        .file("src/ffi.cpp")
        .compile("ffi");

    println!("cargo:rerun-if-changed=src/ffi.cpp");
}
