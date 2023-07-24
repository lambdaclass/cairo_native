use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::extensions::core::{CoreLibfunc, CoreType};
use cairo_native::easy::compile_and_execute;
use num_bigint::BigUint;
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

    let program = cairo_lang_compiler::compile_cairo_project_at_path(
        Path::new("programs/examples/hello.cairo"),
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let name = {
        let mut digits = BigUint::from(u32::from_le_bytes(*b"user")).to_u32_digits();
        digits.resize(8, 0);
        digits
    };

    compile_and_execute::<CoreType, CoreLibfunc, _, _>(
        &program,
        &program
            .funcs
            .iter()
            .find(|x| x.id.debug_name.as_deref() == Some("hello::hello::greet"))
            .unwrap()
            .id,
        json!([name]),
        &mut serde_json::Serializer::new(stdout()),
    )
    .unwrap();
    println!();
}
