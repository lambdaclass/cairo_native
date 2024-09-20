use crate::common::load_cairo;
use cairo_native::context::NativeContext;
use std::error::Error;
use tempfile::NamedTempFile;

#[test]
pub fn compile_library() -> Result<(), Box<dyn Error>> {
    // Load the program.
    let context = NativeContext::new();

    let program = load_cairo! {
        fn run_test(lhs: felt252, rhs: felt252) -> felt252 {
            lhs + rhs
        }
    };

    let module = context.compile(&program.1, false)?;

    let object = cairo_native::module_to_object(module.module(), Default::default())?;

    let file = NamedTempFile::new()?.into_temp_path();
    cairo_native::object_to_shared_lib(&object, &file)?;

    Ok(())
}
