fn main() {
    cxx_build::bridge("src/ffi.rs")
        .file("cpp/mlir.cpp")
        .flag("-std=c++17")
        .flag("-I/usr/lib/llvm-16/include/")
        .flag("-Wno-unused-parameter")
        .compile("sierra2mlir-ffi");

    println!("cargo:rerun-if-changed=cpp/mlir.hpp");
    println!("cargo:rerun-if-changed=cpp/mlir.cpp");
    println!("cargo:rerun-if-changed=src/ffi.rs");

    println!("cargo:rustc-link-search=/usr/lib/llvm-16/lib/");
    println!("cargo:rustc-link-lib=LLVM");
    println!("cargo:rustc-link-lib=MLIR");
}
