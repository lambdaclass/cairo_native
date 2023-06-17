fn main() {
    cc::Build::new()
        .cpp(true)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-I/usr/lib/llvm-16/include")
        .flag_if_supported("-Wno-unused-parameter")
        .file("src/ffi.cpp")
        .compile("ffi");

    println!("cargo:rerun-if-changed=src/ffi.cpp");
}
