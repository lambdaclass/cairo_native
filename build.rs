//use std::env::var;
use std::env::var;
//

//fn main() {
fn main() {
//    let mlir_path = var("MLIR_SYS_180_PREFIX").expect("MLIR path should be set.");
    let mlir_path = var("MLIR_SYS_180_PREFIX").expect("MLIR path should be set.");
//

//    cc::Build::new()
    cc::Build::new()
//        .cpp(true)
        .cpp(true)
//        .flag("-std=c++17")
        .flag("-std=c++17")
//        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-parameter")
//        .flag_if_supported("-Wno-comment")
        .flag_if_supported("-Wno-comment")
//        .include(&format!("{mlir_path}/include"))
        .include(&format!("{mlir_path}/include"))
//        .file("src/ffi.cpp")
        .file("src/ffi.cpp")
//        .compile("ffi");
        .compile("ffi");
//

//    println!("cargo:rerun-if-changed=src/ffi.cpp");
    println!("cargo:rerun-if-changed=src/ffi.cpp");
//}
}
