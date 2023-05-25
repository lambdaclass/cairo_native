//! Utility functions.

use crate::mlir_sys::{
    mlirParsePassPipeline, mlirRegisterAllDialects, mlirRegisterAllLLVMTranslations,
    mlirRegisterAllPasses, MlirStringRef,
};
use crate::{
    context::Context, dialect, logical_result::LogicalResult, pass, string_ref::StringRef, Error,
};
use std::{
    ffi::c_void,
    fmt::{self, Formatter},
    sync::Once,
};

/// Registers all dialects to a dialect registry.
pub fn register_all_dialects(registry: &dialect::Registry) {
    unsafe { mlirRegisterAllDialects(registry.to_raw()) }
}

/// Register all translations from other dialects to the `llvm` dialect.
pub fn register_all_llvm_translations(context: &Context) {
    unsafe { mlirRegisterAllLLVMTranslations(context.to_raw()) }
}

/// Register all passes.
pub fn register_all_passes() {
    static ONCE: Once = Once::new();

    // Multiple calls of `mlirRegisterAllPasses` seems to cause double free.
    ONCE.call_once(|| unsafe { mlirRegisterAllPasses() });
}

/// Parses a pass pipeline.
pub fn parse_pass_pipeline(manager: pass::OperationManager, source: &str) -> Result<(), Error> {
    let mut error_msg = String::new();
    let result = LogicalResult::from_raw(unsafe {
        mlirParsePassPipeline(
            manager.to_raw(),
            StringRef::from(source).to_raw(),
            Some(error_callback),
            &mut error_msg as *mut _ as *mut c_void,
        )
    });

    if result.is_success() {
        Ok(())
    } else {
        Err(Error::ParsePassPipeline(error_msg))
    }
}

// TODO Use into_raw_parts.
pub(crate) unsafe fn into_raw_array<T>(mut xs: Vec<T>) -> *mut T {
    xs.shrink_to_fit();
    xs.leak().as_mut_ptr()
}

pub(crate) unsafe extern "C" fn print_callback(string: MlirStringRef, data: *mut c_void) {
    let (formatter, result) = &mut *(data as *mut (&mut Formatter, fmt::Result));

    if result.is_err() {
        return;
    }

    *result = (|| -> fmt::Result {
        write!(formatter, "{}", StringRef::from_raw(string).as_str().map_err(|_| fmt::Error)?)
    })();
}

pub(crate) unsafe extern "C" fn print_callback_v2(string: MlirStringRef, data: *mut c_void) {
    let data = &mut *(data as *mut String);

    let debug = StringRef::from_raw(string);
    let debug_str = debug.as_str().unwrap();
    data.push_str(debug_str);
}

// used to print ops with debug info
pub(crate) unsafe extern "C" fn print_debug_callback(string: MlirStringRef, data: *mut c_void) {
    let data = &mut *(data as *mut String);

    let debug = StringRef::from_raw(string);
    let debug_str = debug.as_str().unwrap();
    data.push_str(debug_str);
}

pub(crate) unsafe extern "C" fn error_callback(string: MlirStringRef, data: *mut c_void) {
    let data = &mut *(data as *mut String);

    let err = StringRef::from_raw(string);
    let error = err.as_str().unwrap();
    data.push_str(error);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_dialects() {
        let registry = dialect::Registry::new();

        register_all_dialects(&registry);
    }

    #[test]
    fn register_dialects_twice() {
        let registry = dialect::Registry::new();

        register_all_dialects(&registry);
        register_all_dialects(&registry);
    }

    #[test]
    fn register_llvm_translations() {
        let context = Context::new();

        register_all_llvm_translations(&context);
    }

    #[test]
    fn register_llvm_translations_twice() {
        let context = Context::new();

        register_all_llvm_translations(&context);
        register_all_llvm_translations(&context);
    }

    #[test]
    fn register_passes() {
        register_all_passes();
    }

    #[test]
    fn register_passes_twice() {
        register_all_passes();
        register_all_passes();
    }

    #[test]
    fn register_passes_many_times() {
        for _ in 0..1000 {
            register_all_passes();
        }
    }
}
