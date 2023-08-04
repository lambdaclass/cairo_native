use cairo_native::easy::{compile_and_execute, felt252_short_str};
use serde_json::json;
use std::{io::stdout, path::Path};

fn main() {
    // FIXME: Remove when cairo adds an easy to use API for setting the corelibs path.
    std::env::set_var(
        "CARGO_MANIFEST_DIR",
        format!("{}/a", std::env::var("CARGO_MANIFEST_DIR").unwrap()),
    );

    #[cfg(not(feature = "with-runtime"))]
    compile_error!("This example requires the `with-runtime` feature to be active.");

    let name = felt252_short_str("user");

    // Compile and execute the given sierra program, with the inputs and outputs serialized using JSON.
    compile_and_execute(
        Path::new("programs/examples/hello.cairo"),
        "hello::hello::greet",
        json!([name]),
        &mut serde_json::Serializer::new(stdout()),
    )
    .unwrap();
    println!();
}
