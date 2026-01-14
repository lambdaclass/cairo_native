use crate::common::get_compiled_program;
use cairo_native::context::NativeContext;
use std::error::Error;
use tempfile::NamedTempFile;

#[test]
pub fn compile_library() -> Result<(), Box<dyn Error>> {
    // Load the program.
    let context = NativeContext::new();

    let program = get_compiled_program("test_data_artifacts/programs/felt252_add");

    let module = context.compile(&program.1, false, Some(Default::default()), None)?;

    let object = cairo_native::module_to_object(module.module(), Default::default(), None)?;

    let file = NamedTempFile::new()?.into_temp_path();
    cairo_native::object_to_shared_lib(&object, &file, None)?;

    Ok(())
}
