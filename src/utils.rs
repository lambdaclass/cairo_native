////! # Various utilities
//! # Various utilities
//

//use crate::{
use crate::{
//    debug_info::{DebugInfo, DebugLocations},
    debug_info::{DebugInfo, DebugLocations},
//    metadata::MetadataStorage,
    metadata::MetadataStorage,
//    types::{felt252::PRIME, TypeBuilder},
    types::{felt252::PRIME, TypeBuilder},
//    OptLevel,
    OptLevel,
//};
};
//use cairo_lang_compiler::{
use cairo_lang_compiler::{
//    compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
    compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
//    project::setup_project, CompilerConfig,
    project::setup_project, CompilerConfig,
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::core::{CoreLibfunc, CoreType},
    extensions::core::{CoreLibfunc, CoreType},
//    ids::{ConcreteTypeId, FunctionId},
    ids::{ConcreteTypeId, FunctionId},
//    program::{GenFunction, Program, StatementIdx},
    program::{GenFunction, Program, StatementIdx},
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    ir::{Module, Type},
    ir::{Module, Type},
//    pass::{self, PassManager},
    pass::{self, PassManager},
//    Context, Error, ExecutionEngine,
    Context, Error, ExecutionEngine,
//};
};
//use num_bigint::{BigInt, BigUint, Sign};
use num_bigint::{BigInt, BigUint, Sign};
//use std::{
use std::{
//    alloc::Layout,
    alloc::Layout,
//    borrow::Cow,
    borrow::Cow,
//    fmt::{self, Display},
    fmt::{self, Display},
//    ops::Neg,
    ops::Neg,
//    path::Path,
    path::Path,
//    ptr::NonNull,
    ptr::NonNull,
//    sync::Arc,
    sync::Arc,
//};
};
//use thiserror::Error;
use thiserror::Error;
//

//#[cfg(target_os = "macos")]
#[cfg(target_os = "macos")]
//pub const SHARED_LIBRARY_EXT: &str = "dylib";
pub const SHARED_LIBRARY_EXT: &str = "dylib";
//#[cfg(target_os = "linux")]
#[cfg(target_os = "linux")]
//pub const SHARED_LIBRARY_EXT: &str = "so";
pub const SHARED_LIBRARY_EXT: &str = "so";
//

///// Generate a function name.
/// Generate a function name.
/////
///
///// If the program includes function identifiers, return those. Otherwise return `f` followed by the
/// If the program includes function identifiers, return those. Otherwise return `f` followed by the
///// identifier number.
/// identifier number.
//pub fn generate_function_name(function_id: &FunctionId) -> Cow<str> {
pub fn generate_function_name(function_id: &FunctionId) -> Cow<str> {
//    // Generic functions can omit their type in the debug_name, leading to multiple functions
    // Generic functions can omit their type in the debug_name, leading to multiple functions
//    // having the same name, we solve this by adding the id number even if the function has a debug_name
    // having the same name, we solve this by adding the id number even if the function has a debug_name
//    if let Some(name) = function_id.debug_name.as_deref() {
    if let Some(name) = function_id.debug_name.as_deref() {
//        Cow::Owned(format!("{}(f{})", name, function_id.id))
        Cow::Owned(format!("{}(f{})", name, function_id.id))
//    } else {
    } else {
//        Cow::Owned(format!("f{}", function_id.id))
        Cow::Owned(format!("f{}", function_id.id))
//    }
    }
//}
}
//

///// Return the layout for an integer of arbitrary width.
/// Return the layout for an integer of arbitrary width.
/////
///
///// This assumes the platform's maximum (effective) alignment is 16 bytes, and that every integer
/// This assumes the platform's maximum (effective) alignment is 16 bytes, and that every integer
///// with a size in bytes of a power of two has the same alignment as its size.
/// with a size in bytes of a power of two has the same alignment as its size.
//pub fn get_integer_layout(width: u32) -> Layout {
pub fn get_integer_layout(width: u32) -> Layout {
//    if width == 0 {
    if width == 0 {
//        Layout::new::<()>()
        Layout::new::<()>()
//    } else if width <= 8 {
    } else if width <= 8 {
//        Layout::new::<u8>()
        Layout::new::<u8>()
//    } else if width <= 16 {
    } else if width <= 16 {
//        Layout::new::<u16>()
        Layout::new::<u16>()
//    } else if width <= 32 {
    } else if width <= 32 {
//        Layout::new::<u32>()
        Layout::new::<u32>()
//    } else if width <= 64 {
    } else if width <= 64 {
//        Layout::new::<u64>()
        Layout::new::<u64>()
//    } else if width <= 128 {
    } else if width <= 128 {
//        #[cfg(not(target_arch = "x86_64"))]
        #[cfg(not(target_arch = "x86_64"))]
//        {
        {
//            Layout::new::<u128>()
            Layout::new::<u128>()
//        }
        }
//        #[cfg(target_arch = "x86_64")]
        #[cfg(target_arch = "x86_64")]
//        {
        {
//            Layout::new::<u128>().align_to(16).unwrap()
            Layout::new::<u128>().align_to(16).unwrap()
//        }
        }
//    } else {
    } else {
//        let width = (width as usize).next_multiple_of(8).next_power_of_two();
        let width = (width as usize).next_multiple_of(8).next_power_of_two();
//        Layout::from_size_align(width >> 3, (width >> 3).min(16)).unwrap()
        Layout::from_size_align(width >> 3, (width >> 3).min(16)).unwrap()
//    }
    }
//}
}
//

///// Compile a cairo program found at the given path to sierra.
/// Compile a cairo program found at the given path to sierra.
//pub fn cairo_to_sierra(program: &Path) -> Arc<Program> {
pub fn cairo_to_sierra(program: &Path) -> Arc<Program> {
//    if program
    if program
//        .extension()
        .extension()
//        .map(|x| {
        .map(|x| {
//            x.to_ascii_lowercase()
            x.to_ascii_lowercase()
//                .to_string_lossy()
                .to_string_lossy()
//                .eq_ignore_ascii_case("cairo")
                .eq_ignore_ascii_case("cairo")
//        })
        })
//        .unwrap_or(false)
        .unwrap_or(false)
//    {
    {
//        cairo_lang_compiler::compile_cairo_project_at_path(
        cairo_lang_compiler::compile_cairo_project_at_path(
//            program,
            program,
//            CompilerConfig {
            CompilerConfig {
//                replace_ids: true,
                replace_ids: true,
//                ..Default::default()
                ..Default::default()
//            },
            },
//        )
        )
//        .unwrap()
        .unwrap()
//        .into()
        .into()
//    } else {
    } else {
//        let source = std::fs::read_to_string(program).unwrap();
        let source = std::fs::read_to_string(program).unwrap();
//        cairo_lang_sierra::ProgramParser::new()
        cairo_lang_sierra::ProgramParser::new()
//            .parse(&source)
            .parse(&source)
//            .unwrap()
            .unwrap()
//            .into()
            .into()
//    }
    }
//}
}
//

//pub fn cairo_to_sierra_with_debug_info<'ctx>(
pub fn cairo_to_sierra_with_debug_info<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    program: &Path,
    program: &Path,
//) -> Result<(Program, DebugLocations<'ctx>), crate::error::Error> {
) -> Result<(Program, DebugLocations<'ctx>), crate::error::Error> {
//    let mut db = RootDatabase::builder().detect_corelib().build().unwrap();
    let mut db = RootDatabase::builder().detect_corelib().build().unwrap();
//    let main_crate_ids = setup_project(&mut db, program).unwrap();
    let main_crate_ids = setup_project(&mut db, program).unwrap();
//    let program = compile_prepared_db(
    let program = compile_prepared_db(
//        &mut db,
        &mut db,
//        main_crate_ids,
        main_crate_ids,
//        CompilerConfig {
        CompilerConfig {
//            replace_ids: true,
            replace_ids: true,
//            ..Default::default()
            ..Default::default()
//        },
        },
//    )
    )
//    .unwrap();
    .unwrap();
//

//    let debug_locations = {
    let debug_locations = {
//        let debug_info = DebugInfo::extract(&db, &program)
        let debug_info = DebugInfo::extract(&db, &program)
//            .map_err(|_| {
            .map_err(|_| {
//                let mut buffer = String::new();
                let mut buffer = String::new();
//                assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
                assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
//                buffer
                buffer
//            })
            })
//            .unwrap();
            .unwrap();
//

//        DebugLocations::extract(context, &db, &debug_info)
        DebugLocations::extract(context, &db, &debug_info)
//    };
    };
//

//    Ok((program, debug_locations))
    Ok((program, debug_locations))
//}
}
//

///// Returns the given entry point if present.
/// Returns the given entry point if present.
//pub fn find_entry_point<'a>(
pub fn find_entry_point<'a>(
//    program: &'a Program,
    program: &'a Program,
//    entry_point: &str,
    entry_point: &str,
//) -> Option<&'a GenFunction<StatementIdx>> {
) -> Option<&'a GenFunction<StatementIdx>> {
//    program
    program
//        .funcs
        .funcs
//        .iter()
        .iter()
//        .find(|x| x.id.debug_name.as_deref() == Some(entry_point))
        .find(|x| x.id.debug_name.as_deref() == Some(entry_point))
//}
}
//

///// Returns the given entry point if present.
/// Returns the given entry point if present.
//pub fn find_entry_point_by_idx(
pub fn find_entry_point_by_idx(
//    program: &Program,
    program: &Program,
//    entry_point_idx: usize,
    entry_point_idx: usize,
//) -> Option<&GenFunction<StatementIdx>> {
) -> Option<&GenFunction<StatementIdx>> {
//    program
    program
//        .funcs
        .funcs
//        .iter()
        .iter()
//        .find(|x| x.id.id == entry_point_idx as u64)
        .find(|x| x.id.id == entry_point_idx as u64)
//}
}
//

///// Given a string representing a function name, searches in the program for the id corresponding to said function, and returns a reference to it.
/// Given a string representing a function name, searches in the program for the id corresponding to said function, and returns a reference to it.
//#[track_caller]
#[track_caller]
//pub fn find_function_id<'a>(program: &'a Program, function_name: &str) -> &'a FunctionId {
pub fn find_function_id<'a>(program: &'a Program, function_name: &str) -> &'a FunctionId {
//    &program
    &program
//        .funcs
        .funcs
//        .iter()
        .iter()
//        .find(|x| x.id.debug_name.as_deref() == Some(function_name))
        .find(|x| x.id.debug_name.as_deref() == Some(function_name))
//        .unwrap()
        .unwrap()
//        .id
        .id
//}
}
//

///// Parse a numeric string into felt, wrapping negatives around the prime modulo.
/// Parse a numeric string into felt, wrapping negatives around the prime modulo.
//pub fn felt252_str(value: &str) -> [u32; 8] {
pub fn felt252_str(value: &str) -> [u32; 8] {
//    let value = value
    let value = value
//        .parse::<BigInt>()
        .parse::<BigInt>()
//        .expect("value must be a digit number");
        .expect("value must be a digit number");
//    let value = match value.sign() {
    let value = match value.sign() {
//        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
//        _ => value.to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
//    };
    };
//

//    let mut u32_digits = value.to_u32_digits();
    let mut u32_digits = value.to_u32_digits();
//    u32_digits.resize(8, 0);
    u32_digits.resize(8, 0);
//    u32_digits.try_into().unwrap()
    u32_digits.try_into().unwrap()
//}
}
//

///// Parse any type that can be a bigint to a felt that can be used in the cairo-native input.
/// Parse any type that can be a bigint to a felt that can be used in the cairo-native input.
//pub fn felt252_bigint(value: impl Into<BigInt>) -> [u32; 8] {
pub fn felt252_bigint(value: impl Into<BigInt>) -> [u32; 8] {
//    let value: BigInt = value.into();
    let value: BigInt = value.into();
//    let value = match value.sign() {
    let value = match value.sign() {
//        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
//        _ => value.to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
//    };
    };
//

//    let mut u32_digits = value.to_u32_digits();
    let mut u32_digits = value.to_u32_digits();
//    u32_digits.resize(8, 0);
    u32_digits.resize(8, 0);
//    u32_digits.try_into().unwrap()
    u32_digits.try_into().unwrap()
//}
}
//

///// Parse a short string into a felt that can be used in the cairo-native input.
/// Parse a short string into a felt that can be used in the cairo-native input.
//pub fn felt252_short_str(value: &str) -> [u32; 8] {
pub fn felt252_short_str(value: &str) -> [u32; 8] {
//    let values: Vec<_> = value
    let values: Vec<_> = value
//        .chars()
        .chars()
//        .filter(|&c| c.is_ascii())
        .filter(|&c| c.is_ascii())
//        .map(|c| c as u8)
        .map(|c| c as u8)
//        .collect();
        .collect();
//

//    let mut digits = BigUint::from_bytes_be(&values).to_u32_digits();
    let mut digits = BigUint::from_bytes_be(&values).to_u32_digits();
//    digits.resize(8, 0);
    digits.resize(8, 0);
//    digits.try_into().unwrap()
    digits.try_into().unwrap()
//}
}
//

///// Creates the execution engine, with all symbols registered.
/// Creates the execution engine, with all symbols registered.
//pub fn create_engine(
pub fn create_engine(
//    module: &Module,
    module: &Module,
//    _metadata: &MetadataStorage,
    _metadata: &MetadataStorage,
//    opt_level: OptLevel,
    opt_level: OptLevel,
//) -> ExecutionEngine {
) -> ExecutionEngine {
//    // Create the JIT engine.
    // Create the JIT engine.
//    let engine = ExecutionEngine::new(module, opt_level.into(), &[], false);
    let engine = ExecutionEngine::new(module, opt_level.into(), &[], false);
//

//    #[cfg(feature = "with-runtime")]
    #[cfg(feature = "with-runtime")]
//    register_runtime_symbols(&engine);
    register_runtime_symbols(&engine);
//

//    #[cfg(feature = "with-debug-utils")]
    #[cfg(feature = "with-debug-utils")]
//    _metadata
    _metadata
//        .get::<crate::metadata::debug_utils::DebugUtils>()
        .get::<crate::metadata::debug_utils::DebugUtils>()
//        .unwrap()
        .unwrap()
//        .register_impls(&engine);
        .register_impls(&engine);
//

//    engine
    engine
//}
}
//

//pub fn run_pass_manager(context: &Context, module: &mut Module) -> Result<(), Error> {
pub fn run_pass_manager(context: &Context, module: &mut Module) -> Result<(), Error> {
//    let pass_manager = PassManager::new(context);
    let pass_manager = PassManager::new(context);
//    pass_manager.enable_verifier(true);
    pass_manager.enable_verifier(true);
//    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::transform::create_canonicalizer());
//    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
//    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
//    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
//    pass_manager.add_pass(pass::conversion::create_index_to_llvm());
    pass_manager.add_pass(pass::conversion::create_index_to_llvm());
//    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
//    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
//    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
//    pass_manager.run(module)
    pass_manager.run(module)
//}
}
//

//#[cfg(feature = "with-runtime")]
#[cfg(feature = "with-runtime")]
//pub fn register_runtime_symbols(engine: &ExecutionEngine) {
pub fn register_runtime_symbols(engine: &ExecutionEngine) {
//    unsafe {
    unsafe {
//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__libfunc__debug__print",
            "cairo_native__libfunc__debug__print",
//            cairo_native_runtime::cairo_native__libfunc__debug__print
            cairo_native_runtime::cairo_native__libfunc__debug__print
//                as *const fn(i32, *const [u8; 32], usize) -> i32 as *mut (),
                as *const fn(i32, *const [u8; 32], usize) -> i32 as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__libfunc__pedersen",
            "cairo_native__libfunc__pedersen",
//            cairo_native_runtime::cairo_native__libfunc__pedersen
            cairo_native_runtime::cairo_native__libfunc__pedersen
//                as *const fn(*mut u8, *mut u8, *mut u8) -> () as *mut (),
                as *const fn(*mut u8, *mut u8, *mut u8) -> () as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__libfunc__hades_permutation",
            "cairo_native__libfunc__hades_permutation",
//            cairo_native_runtime::cairo_native__libfunc__hades_permutation
            cairo_native_runtime::cairo_native__libfunc__hades_permutation
//                as *const fn(*mut u8, *mut u8, *mut u8) -> () as *mut (),
                as *const fn(*mut u8, *mut u8, *mut u8) -> () as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__libfunc__ec__ec_point_from_x_nz",
            "cairo_native__libfunc__ec__ec_point_from_x_nz",
//            cairo_native_runtime::cairo_native__libfunc__ec__ec_point_from_x_nz
            cairo_native_runtime::cairo_native__libfunc__ec__ec_point_from_x_nz
//                as *const fn(*mut [[u8; 32]; 2]) -> bool as *mut (),
                as *const fn(*mut [[u8; 32]; 2]) -> bool as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__libfunc__ec__ec_state_add",
            "cairo_native__libfunc__ec__ec_state_add",
//            cairo_native_runtime::cairo_native__libfunc__ec__ec_state_add
            cairo_native_runtime::cairo_native__libfunc__ec__ec_state_add
//                as *const fn(*mut [[u8; 32]; 4], *const [[u8; 32]; 2]) -> bool
                as *const fn(*mut [[u8; 32]; 4], *const [[u8; 32]; 2]) -> bool
//                as *mut (),
                as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__libfunc__ec__ec_state_add_mul",
            "cairo_native__libfunc__ec__ec_state_add_mul",
//            cairo_native_runtime::cairo_native__libfunc__ec__ec_state_add_mul
            cairo_native_runtime::cairo_native__libfunc__ec__ec_state_add_mul
//                as *const fn(*mut [[u8; 32]; 4], *const [u8; 32], *const [[u8; 32]; 2]) -> bool
                as *const fn(*mut [[u8; 32]; 4], *const [u8; 32], *const [[u8; 32]; 2]) -> bool
//                as *mut (),
                as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__libfunc__ec__ec_state_try_finalize_nz",
            "cairo_native__libfunc__ec__ec_state_try_finalize_nz",
//            cairo_native_runtime::cairo_native__libfunc__ec__ec_state_try_finalize_nz
            cairo_native_runtime::cairo_native__libfunc__ec__ec_state_try_finalize_nz
//                as *const fn(*const [[u8; 32]; 2], *mut [[u8; 32]; 4]) -> bool
                as *const fn(*const [[u8; 32]; 2], *mut [[u8; 32]; 4]) -> bool
//                as *mut (),
                as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__libfunc__ec__ec_point_try_new_nz",
            "cairo_native__libfunc__ec__ec_point_try_new_nz",
//            cairo_native_runtime::cairo_native__libfunc__ec__ec_point_try_new_nz
            cairo_native_runtime::cairo_native__libfunc__ec__ec_point_try_new_nz
//                as *const fn(*const [[u8; 32]; 2]) -> bool as *mut (),
                as *const fn(*const [[u8; 32]; 2]) -> bool as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__alloc_dict",
            "cairo_native__alloc_dict",
//            cairo_native_runtime::cairo_native__alloc_dict as *const fn() -> *mut std::ffi::c_void
            cairo_native_runtime::cairo_native__alloc_dict as *const fn() -> *mut std::ffi::c_void
//                as *mut (),
                as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__dict_free",
            "cairo_native__dict_free",
//            cairo_native_runtime::cairo_native__dict_free as *const fn(*mut std::ffi::c_void) -> ()
            cairo_native_runtime::cairo_native__dict_free as *const fn(*mut std::ffi::c_void) -> ()
//                as *mut (),
                as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__dict_get",
            "cairo_native__dict_get",
//            cairo_native_runtime::cairo_native__dict_get
            cairo_native_runtime::cairo_native__dict_get
//                as *const fn(*mut std::ffi::c_void, &[u8; 32]) -> *mut std::ffi::c_void
                as *const fn(*mut std::ffi::c_void, &[u8; 32]) -> *mut std::ffi::c_void
//                as *mut (),
                as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__dict_insert",
            "cairo_native__dict_insert",
//            cairo_native_runtime::cairo_native__dict_insert
            cairo_native_runtime::cairo_native__dict_insert
//                as *const fn(
                as *const fn(
//                    *mut std::ffi::c_void,
                    *mut std::ffi::c_void,
//                    &[u8; 32],
                    &[u8; 32],
//                    NonNull<std::ffi::c_void>,
                    NonNull<std::ffi::c_void>,
//                ) -> *mut std::ffi::c_void as *mut (),
                ) -> *mut std::ffi::c_void as *mut (),
//        );
        );
//

//        engine.register_symbol(
        engine.register_symbol(
//            "cairo_native__dict_gas_refund",
            "cairo_native__dict_gas_refund",
//            cairo_native_runtime::cairo_native__dict_gas_refund
            cairo_native_runtime::cairo_native__dict_gas_refund
//                as *const fn(*const std::ffi::c_void, NonNull<std::ffi::c_void>) -> u64
                as *const fn(*const std::ffi::c_void, NonNull<std::ffi::c_void>) -> u64
//                as *mut (),
                as *mut (),
//        );
        );
//    }
    }
//}
}
//

///// Return a type that calls a closure when formatted using [Debug](std::fmt::Debug).
/// Return a type that calls a closure when formatted using [Debug](std::fmt::Debug).
//pub fn debug_with<F>(fmt: F) -> impl fmt::Debug
pub fn debug_with<F>(fmt: F) -> impl fmt::Debug
//where
where
//    F: Fn(&mut fmt::Formatter) -> fmt::Result,
    F: Fn(&mut fmt::Formatter) -> fmt::Result,
//{
{
//    struct FmtWrapper<F>(F)
    struct FmtWrapper<F>(F)
//    where
    where
//        F: Fn(&mut fmt::Formatter) -> fmt::Result;
        F: Fn(&mut fmt::Formatter) -> fmt::Result;
//

//    impl<F> fmt::Debug for FmtWrapper<F>
    impl<F> fmt::Debug for FmtWrapper<F>
//    where
    where
//        F: Fn(&mut fmt::Formatter) -> fmt::Result,
        F: Fn(&mut fmt::Formatter) -> fmt::Result,
//    {
    {
//        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//            self.0(f)
            self.0(f)
//        }
        }
//    }
    }
//

//    FmtWrapper(fmt)
    FmtWrapper(fmt)
//}
}
//

//// POLYFILLS of nightly features
// POLYFILLS of nightly features
//

//#[inline]
#[inline]
//pub const fn next_multiple_of_usize(lhs: usize, rhs: usize) -> usize {
pub const fn next_multiple_of_usize(lhs: usize, rhs: usize) -> usize {
//    match lhs % rhs {
    match lhs % rhs {
//        0 => lhs,
        0 => lhs,
//        r => lhs + (rhs - r),
        r => lhs + (rhs - r),
//    }
    }
//}
}
//

//#[inline]
#[inline]
//pub const fn next_multiple_of_u32(lhs: u32, rhs: u32) -> u32 {
pub const fn next_multiple_of_u32(lhs: u32, rhs: u32) -> u32 {
//    match lhs % rhs {
    match lhs % rhs {
//        0 => lhs,
        0 => lhs,
//        r => lhs + (rhs - r),
        r => lhs + (rhs - r),
//    }
    }
//}
}
//

///// Edit: Copied from the std lib.
/// Edit: Copied from the std lib.
/////
///
///// Returns the amount of padding we must insert after `layout`
/// Returns the amount of padding we must insert after `layout`
///// to ensure that the following address will satisfy `align`
/// to ensure that the following address will satisfy `align`
///// (measured in bytes).
/// (measured in bytes).
/////
///
///// e.g., if `layout.size()` is 9, then `layout.padding_needed_for(4)`
/// e.g., if `layout.size()` is 9, then `layout.padding_needed_for(4)`
///// returns 3, because that is the minimum number of bytes of
/// returns 3, because that is the minimum number of bytes of
///// padding required to get a 4-aligned address (assuming that the
/// padding required to get a 4-aligned address (assuming that the
///// corresponding memory block starts at a 4-aligned address).
/// corresponding memory block starts at a 4-aligned address).
/////
///
///// The return value of this function has no meaning if `align` is
/// The return value of this function has no meaning if `align` is
///// not a power-of-two.
/// not a power-of-two.
/////
///
///// Note that the utility of the returned value requires `align`
/// Note that the utility of the returned value requires `align`
///// to be less than or equal to the alignment of the starting
/// to be less than or equal to the alignment of the starting
///// address for the whole allocated block of memory. One way to
/// address for the whole allocated block of memory. One way to
///// satisfy this constraint is to ensure `align <= layout.align()`.
/// satisfy this constraint is to ensure `align <= layout.align()`.
//#[inline]
#[inline]
//pub const fn padding_needed_for(layout: &Layout, align: usize) -> usize {
pub const fn padding_needed_for(layout: &Layout, align: usize) -> usize {
//    let len = layout.size();
    let len = layout.size();
//

//    // Rounded up value is:
    // Rounded up value is:
//    //   len_rounded_up = (len + align - 1) & !(align - 1);
    //   len_rounded_up = (len + align - 1) & !(align - 1);
//    // and then we return the padding difference: `len_rounded_up - len`.
    // and then we return the padding difference: `len_rounded_up - len`.
//    //
    //
//    // We use modular arithmetic throughout:
    // We use modular arithmetic throughout:
//    //
    //
//    // 1. align is guaranteed to be > 0, so align - 1 is always
    // 1. align is guaranteed to be > 0, so align - 1 is always
//    //    valid.
    //    valid.
//    //
    //
//    // 2. `len + align - 1` can overflow by at most `align - 1`,
    // 2. `len + align - 1` can overflow by at most `align - 1`,
//    //    so the &-mask with `!(align - 1)` will ensure that in the
    //    so the &-mask with `!(align - 1)` will ensure that in the
//    //    case of overflow, `len_rounded_up` will itself be 0.
    //    case of overflow, `len_rounded_up` will itself be 0.
//    //    Thus the returned padding, when added to `len`, yields 0,
    //    Thus the returned padding, when added to `len`, yields 0,
//    //    which trivially satisfies the alignment `align`.
    //    which trivially satisfies the alignment `align`.
//    //
    //
//    // (Of course, attempts to allocate blocks of memory whose
    // (Of course, attempts to allocate blocks of memory whose
//    // size and padding overflow in the above manner should cause
    // size and padding overflow in the above manner should cause
//    // the allocator to yield an error anyway.)
    // the allocator to yield an error anyway.)
//

//    let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);
    let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);
//    len_rounded_up.wrapping_sub(len)
    len_rounded_up.wrapping_sub(len)
//}
}
//

//#[derive(Clone, PartialEq, Eq, Debug, Error)]
#[derive(Clone, PartialEq, Eq, Debug, Error)]
//pub struct LayoutError;
pub struct LayoutError;
//

//impl Display for LayoutError {
impl Display for LayoutError {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        f.write_str("layout error")
        f.write_str("layout error")
//    }
    }
//}
}
//

///// Copied from std.
/// Copied from std.
/////
///
///// Creates a layout describing the record for `n` instances of
/// Creates a layout describing the record for `n` instances of
///// `self`, with a suitable amount of padding between each to
/// `self`, with a suitable amount of padding between each to
///// ensure that each instance is given its requested size and
/// ensure that each instance is given its requested size and
///// alignment. On success, returns `(k, offs)` where `k` is the
/// alignment. On success, returns `(k, offs)` where `k` is the
///// layout of the array and `offs` is the distance between the start
/// layout of the array and `offs` is the distance between the start
///// of each element in the array.
/// of each element in the array.
/////
///
///// On arithmetic overflow, returns `LayoutError`.
/// On arithmetic overflow, returns `LayoutError`.
////#[unstable(feature = "alloc_layout_extra", issue = "55724")]
//#[unstable(feature = "alloc_layout_extra", issue = "55724")]
//#[inline]
#[inline]
//pub fn layout_repeat(layout: &Layout, n: usize) -> Result<(Layout, usize), LayoutError> {
pub fn layout_repeat(layout: &Layout, n: usize) -> Result<(Layout, usize), LayoutError> {
//    // This cannot overflow. Quoting from the invariant of Layout:
    // This cannot overflow. Quoting from the invariant of Layout:
//    // > `size`, when rounded up to the nearest multiple of `align`,
    // > `size`, when rounded up to the nearest multiple of `align`,
//    // > must not overflow isize (i.e., the rounded value must be
    // > must not overflow isize (i.e., the rounded value must be
//    // > less than or equal to `isize::MAX`)
    // > less than or equal to `isize::MAX`)
//    let padded_size = layout.size() + padding_needed_for(layout, layout.align());
    let padded_size = layout.size() + padding_needed_for(layout, layout.align());
//    let alloc_size = padded_size.checked_mul(n).ok_or(LayoutError)?;
    let alloc_size = padded_size.checked_mul(n).ok_or(LayoutError)?;
//

//    // The safe constructor is called here to enforce the isize size limit.
    // The safe constructor is called here to enforce the isize size limit.
//    let layout = Layout::from_size_align(alloc_size, layout.align()).map_err(|_| LayoutError)?;
    let layout = Layout::from_size_align(alloc_size, layout.align()).map_err(|_| LayoutError)?;
//    Ok((layout, padded_size))
    Ok((layout, padded_size))
//}
}
//

//pub trait ProgramRegistryExt {
pub trait ProgramRegistryExt {
//    fn build_type<'ctx>(
    fn build_type<'ctx>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        module: &Module<'ctx>,
        module: &Module<'ctx>,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        metadata: &mut MetadataStorage,
        metadata: &mut MetadataStorage,
//        id: &ConcreteTypeId,
        id: &ConcreteTypeId,
//    ) -> Result<Type<'ctx>, super::error::Error>;
    ) -> Result<Type<'ctx>, super::error::Error>;
//

//    fn build_type_with_layout<'ctx>(
    fn build_type_with_layout<'ctx>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        module: &Module<'ctx>,
        module: &Module<'ctx>,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        metadata: &mut MetadataStorage,
        metadata: &mut MetadataStorage,
//        id: &ConcreteTypeId,
        id: &ConcreteTypeId,
//    ) -> Result<(Type<'ctx>, Layout), super::error::Error>;
    ) -> Result<(Type<'ctx>, Layout), super::error::Error>;
//}
}
//

//impl ProgramRegistryExt for ProgramRegistry<CoreType, CoreLibfunc> {
impl ProgramRegistryExt for ProgramRegistry<CoreType, CoreLibfunc> {
//    fn build_type<'ctx>(
    fn build_type<'ctx>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        module: &Module<'ctx>,
        module: &Module<'ctx>,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        metadata: &mut MetadataStorage,
        metadata: &mut MetadataStorage,
//        id: &ConcreteTypeId,
        id: &ConcreteTypeId,
//    ) -> Result<Type<'ctx>, super::error::Error> {
    ) -> Result<Type<'ctx>, super::error::Error> {
//        registry
        registry
//            .get_type(id)?
            .get_type(id)?
//            .build(context, module, registry, metadata, id)
            .build(context, module, registry, metadata, id)
//    }
    }
//

//    fn build_type_with_layout<'ctx>(
    fn build_type_with_layout<'ctx>(
//        &self,
        &self,
//        context: &'ctx Context,
        context: &'ctx Context,
//        module: &Module<'ctx>,
        module: &Module<'ctx>,
//        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//        metadata: &mut MetadataStorage,
        metadata: &mut MetadataStorage,
//        id: &ConcreteTypeId,
        id: &ConcreteTypeId,
//    ) -> Result<(Type<'ctx>, Layout), super::error::Error> {
    ) -> Result<(Type<'ctx>, Layout), super::error::Error> {
//        let concrete_type = registry.get_type(id)?;
        let concrete_type = registry.get_type(id)?;
//

//        Ok((
        Ok((
//            concrete_type.build(context, module, registry, metadata, id)?,
            concrete_type.build(context, module, registry, metadata, id)?,
//            concrete_type.layout(registry)?,
            concrete_type.layout(registry)?,
//        ))
        ))
//    }
    }
//}
}
//

///// The `mlir_asm!` macro is a shortcut to manually building operations.
/// The `mlir_asm!` macro is a shortcut to manually building operations.
/////
///
///// It works by forwarding the custom DSL code to their respective functions within melior's
/// It works by forwarding the custom DSL code to their respective functions within melior's
///// `OperationBuilder`.
/// `OperationBuilder`.
/////
///
///// The DSL's syntax is similar to that of MLIR, but has some differences, or rather restrictions,
/// The DSL's syntax is similar to that of MLIR, but has some differences, or rather restrictions,
///// due to the way declarative macros work:
/// due to the way declarative macros work:
/////   - All macro invocations need the MLIR context, the target block and the operations' locations.
///   - All macro invocations need the MLIR context, the target block and the operations' locations.
/////   - The operations are defined using a syntax similar to that of MLIR's generic operations, with
///   - The operations are defined using a syntax similar to that of MLIR's generic operations, with
/////     some differences. The results are Rust variables (MLIR values) and the inputs (operands,
///     some differences. The results are Rust variables (MLIR values) and the inputs (operands,
/////     attributes...) are all Rust expressions that evaluate to their respective type.
///     attributes...) are all Rust expressions that evaluate to their respective type.
/////
///
///// Check out the [felt252 libfunc implementations](crate::libfuncs::felt252) for an example on their usage.
/// Check out the [felt252 libfunc implementations](crate::libfuncs::felt252) for an example on their usage.
//macro_rules! mlir_asm {
macro_rules! mlir_asm {
//    (
    (
//        $context:expr, $block:expr, $location:expr =>
        $context:expr, $block:expr, $location:expr =>
//            $( ; $( $( $ret:ident ),+ = )? $op:literal
            $( ; $( $( $ret:ident ),+ = )? $op:literal
//                ( $( $( $arg:expr ),+ $(,)? )? ) // Operands.
                ( $( $( $arg:expr ),+ $(,)? )? ) // Operands.
//                $( [ $( $( ^ $successor:ident $( ( $( $( $successor_arg:expr ),+ $(,)? )? ) )? ),+ $(,)? )? ] )? // Successors.
                $( [ $( $( ^ $successor:ident $( ( $( $( $successor_arg:expr ),+ $(,)? )? ) )? ),+ $(,)? )? ] )? // Successors.
//                $( < { $( $( $prop_name:pat_param = $prop_value:expr ),+ $(,)? )? } > )? // Properties.
                $( < { $( $( $prop_name:pat_param = $prop_value:expr ),+ $(,)? )? } > )? // Properties.
//                $( ( $( $( $region:expr ),+ $(,)? )? ) )? // Regions.
                $( ( $( $( $region:expr ),+ $(,)? )? ) )? // Regions.
//                $( { $( $( $attr_name:literal = $attr_value:expr ),+ $(,)? )? } )? // Attributes.
                $( { $( $( $attr_name:literal = $attr_value:expr ),+ $(,)? )? } )? // Attributes.
//                : $args_ty:tt -> $rets_ty:tt // Signature.
                : $args_ty:tt -> $rets_ty:tt // Signature.
//            )*
            )*
//    ) => { $(
    ) => { $(
//        #[allow(unused_mut)]
        #[allow(unused_mut)]
//        $( let $crate::utils::codegen_ret_decl!($($ret),+) = )? {
        $( let $crate::utils::codegen_ret_decl!($($ret),+) = )? {
//            #[allow(unused_variables)]
            #[allow(unused_variables)]
//            let context = $context;
            let context = $context;
//            let mut builder = melior::ir::operation::OperationBuilder::new($op, $location);
            let mut builder = melior::ir::operation::OperationBuilder::new($op, $location);
//

//            // Process operands.
            // Process operands.
//            $( let builder = builder.add_operands(&[$( $arg, )+]); )?
            $( let builder = builder.add_operands(&[$( $arg, )+]); )?
//

//            // TODO: Process successors.
            // TODO: Process successors.
//            // TODO: Process properties.
            // TODO: Process properties.
//            // TODO: Process regions.
            // TODO: Process regions.
//

//            // Process attributes.
            // Process attributes.
//            $( $(
            $( $(
//                let builder = $crate::utils::codegen_attributes!(context, builder => $($attr_name = $attr_value),+);
                let builder = $crate::utils::codegen_attributes!(context, builder => $($attr_name = $attr_value),+);
//            )? )?
            )? )?
//

//            // Process signature.
            // Process signature.
//            // #[cfg(debug_assertions)]
            // #[cfg(debug_assertions)]
//            // $crate::utils::codegen_signature!( PARAMS $args_ty );
            // $crate::utils::codegen_signature!( PARAMS $args_ty );
//            let builder = $crate::utils::codegen_signature!( RETS builder => $rets_ty );
            let builder = $crate::utils::codegen_signature!( RETS builder => $rets_ty );
//

//            #[allow(unused_variables)]
            #[allow(unused_variables)]
//            let op = $block.append_operation(builder.build()?);
            let op = $block.append_operation(builder.build()?);
//            $( $crate::utils::codegen_ret_extr!(op => $($ret),+) )?
            $( $crate::utils::codegen_ret_extr!(op => $($ret),+) )?
//        };
        };
//    )* };
    )* };
//}
}
//pub(crate) use mlir_asm;
pub(crate) use mlir_asm;
//

//macro_rules! codegen_attributes {
macro_rules! codegen_attributes {
//    // Macro entry points.
    // Macro entry points.
//    ( $context:ident, $builder:ident => $name:literal = $value:expr ) => {
    ( $context:ident, $builder:ident => $name:literal = $value:expr ) => {
//        $builder.add_attributes(&[
        $builder.add_attributes(&[
//            $crate::utils::codegen_attributes!(INTERNAL $context, $builder => $name = $value),
            $crate::utils::codegen_attributes!(INTERNAL $context, $builder => $name = $value),
//        ])
        ])
//    };
    };
//    ( $context:ident, $builder:ident => $( $name:literal = $value:expr ),+ ) => {
    ( $context:ident, $builder:ident => $( $name:literal = $value:expr ),+ ) => {
//        $builder.add_attributes(&[
        $builder.add_attributes(&[
//            $( $crate::utils::codegen_attributes!(INTERNAL $context, $builder => $name = $value), )+
            $( $crate::utils::codegen_attributes!(INTERNAL $context, $builder => $name = $value), )+
//        ])
        ])
//    };
    };
//

//    ( INTERNAL $context:ident, $builder:ident => $name:literal = $value:expr ) => {
    ( INTERNAL $context:ident, $builder:ident => $name:literal = $value:expr ) => {
//        (
        (
//            melior::ir::Identifier::new($context, $name),
            melior::ir::Identifier::new($context, $name),
//            $value,
            $value,
//        )
        )
//    };
    };
//}
}
//pub(crate) use codegen_attributes;
pub(crate) use codegen_attributes;
//

//macro_rules! codegen_signature {
macro_rules! codegen_signature {
//    ( PARAMS ) => {
    ( PARAMS ) => {
//        // TODO: Check operand types.
        // TODO: Check operand types.
//    };
    };
//

//    ( RETS $builder:ident => () ) => { $builder };
    ( RETS $builder:ident => () ) => { $builder };
//    ( RETS $builder:ident => $ret_ty:expr ) => {
    ( RETS $builder:ident => $ret_ty:expr ) => {
//        $builder.add_results(&[$ret_ty])
        $builder.add_results(&[$ret_ty])
//    };
    };
//    ( RETS $builder:ident => $( $ret_ty:expr ),+ $(,)? ) => {
    ( RETS $builder:ident => $( $ret_ty:expr ),+ $(,)? ) => {
//        $builder.add_results(&[$($ret_ty),+])
        $builder.add_results(&[$($ret_ty),+])
//    };
    };
//}
}
//pub(crate) use codegen_signature;
pub(crate) use codegen_signature;
//

//macro_rules! codegen_ret_decl {
macro_rules! codegen_ret_decl {
//    // Macro entry points.
    // Macro entry points.
//    ( $ret:ident ) => { $ret };
    ( $ret:ident ) => { $ret };
//    ( $( $ret:ident ),+ ) => {
    ( $( $ret:ident ),+ ) => {
//        ( $( codegen_ret_decl!($ret) ),+ )
        ( $( codegen_ret_decl!($ret) ),+ )
//    };
    };
//}
}
//pub(crate) use codegen_ret_decl;
pub(crate) use codegen_ret_decl;
//

//macro_rules! codegen_ret_extr {
macro_rules! codegen_ret_extr {
//    // Macro entry points.
    // Macro entry points.
//    ( $op:ident => $ret:ident ) => {{
    ( $op:ident => $ret:ident ) => {{
//        melior::ir::Value::from($op.result(0)?)
        melior::ir::Value::from($op.result(0)?)
//    }};
    }};
//    ( $op:ident => $( $ret:ident ),+ ) => {{
    ( $op:ident => $( $ret:ident ),+ ) => {{
//        let mut idx = 0;
        let mut idx = 0;
//        ( $( codegen_ret_extr!(INTERNAL idx, $op => $ret) ),+ )
        ( $( codegen_ret_extr!(INTERNAL idx, $op => $ret) ),+ )
//    }};
    }};
//

//    // Internal entry points.
    // Internal entry points.
//    ( INTERNAL $count:ident, $op:ident => $ret:ident ) => {
    ( INTERNAL $count:ident, $op:ident => $ret:ident ) => {
//        {
        {
//            let idx = $count;
            let idx = $count;
//            $count += 1;
            $count += 1;
//            melior::ir::Value::from($op.result(idx)?)
            melior::ir::Value::from($op.result(idx)?)
//        }
        }
//    };
    };
//}
}
//pub(crate) use codegen_ret_extr;
pub(crate) use codegen_ret_extr;
//

//#[cfg(test)]
#[cfg(test)]
//pub mod test {
pub mod test {
//    use crate::{
    use crate::{
//        context::NativeContext,
        context::NativeContext,
//        execution_result::ExecutionResult,
        execution_result::ExecutionResult,
//        executor::JitNativeExecutor,
        executor::JitNativeExecutor,
//        starknet::{
        starknet::{
//            BlockInfo, ExecutionInfo, ExecutionInfoV2, ResourceBounds, StarknetSyscallHandler,
            BlockInfo, ExecutionInfo, ExecutionInfoV2, ResourceBounds, StarknetSyscallHandler,
//            SyscallResult, TxInfo, TxV2Info, U256,
            SyscallResult, TxInfo, TxV2Info, U256,
//        },
        },
//        utils::*,
        utils::*,
//        values::JitValue,
        values::JitValue,
//    };
    };
//    use cairo_lang_compiler::{
    use cairo_lang_compiler::{
//        compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
        compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
//        project::setup_project, CompilerConfig,
        project::setup_project, CompilerConfig,
//    };
    };
//    use cairo_lang_filesystem::db::init_dev_corelib;
    use cairo_lang_filesystem::db::init_dev_corelib;
//    use cairo_lang_sierra::{
    use cairo_lang_sierra::{
//        ids::FunctionId,
        ids::FunctionId,
//        program::Program,
        program::Program,
//        program::{FunctionSignature, GenFunction, StatementIdx},
        program::{FunctionSignature, GenFunction, StatementIdx},
//    };
    };
//    use cairo_lang_starknet::starknet_plugin_suite;
    use cairo_lang_starknet::starknet_plugin_suite;
//    use pretty_assertions_sorted::assert_eq;
    use pretty_assertions_sorted::assert_eq;
//    use starknet_types_core::felt::Felt;
    use starknet_types_core::felt::Felt;
//    use std::io::Write;
    use std::io::Write;
//    use std::{env::var, fmt::Formatter, fs, path::Path};
    use std::{env::var, fmt::Formatter, fs, path::Path};
//

//    macro_rules! load_cairo {
    macro_rules! load_cairo {
//        ( $( $program:tt )+ ) => {
        ( $( $program:tt )+ ) => {
//            $crate::utils::test::load_cairo_str(stringify!($($program)+))
            $crate::utils::test::load_cairo_str(stringify!($($program)+))
//        };
        };
//    }
    }
//    macro_rules! load_starknet {
    macro_rules! load_starknet {
//        ( $( $program:tt )+ ) => {
        ( $( $program:tt )+ ) => {
//            $crate::utils::test::load_starknet_str(stringify!($($program)+))
            $crate::utils::test::load_starknet_str(stringify!($($program)+))
//        };
        };
//    }
    }
//    pub(crate) use load_cairo;
    pub(crate) use load_cairo;
//    pub(crate) use load_starknet;
    pub(crate) use load_starknet;
//

//    // Helper macros for faster testing.
    // Helper macros for faster testing.
//    macro_rules! jit_struct {
    macro_rules! jit_struct {
//        ( $($x:expr),* $(,)? ) => {
        ( $($x:expr),* $(,)? ) => {
//            crate::values::JitValue::Struct {
            crate::values::JitValue::Struct {
//                fields: vec![$($x), *],
                fields: vec![$($x), *],
//                debug_name: None
                debug_name: None
//            }
            }
//        };
        };
//    }
    }
//    macro_rules! jit_enum {
    macro_rules! jit_enum {
//        ( $tag:expr, $value:expr ) => {
        ( $tag:expr, $value:expr ) => {
//            crate::values::JitValue::Enum {
            crate::values::JitValue::Enum {
//                tag: $tag,
                tag: $tag,
//                value: Box::new($value),
                value: Box::new($value),
//                debug_name: None,
                debug_name: None,
//            }
            }
//        };
        };
//    }
    }
//    macro_rules! jit_dict {
    macro_rules! jit_dict {
//        ( $($key:expr $(=>)+ $value:expr),* $(,)? ) => {
        ( $($key:expr $(=>)+ $value:expr),* $(,)? ) => {
//            crate::values::JitValue::Felt252Dict {
            crate::values::JitValue::Felt252Dict {
//                value: {
                value: {
//                    let mut map = std::collections::HashMap::new();
                    let mut map = std::collections::HashMap::new();
//                    $(map.insert($key.into(), $value.into());)*
                    $(map.insert($key.into(), $value.into());)*
//                    map
                    map
//                },
                },
//                debug_name: None,
                debug_name: None,
//            }
            }
//        };
        };
//    }
    }
//    macro_rules! jit_panic {
    macro_rules! jit_panic {
//        ( $($value:expr)? ) => {
        ( $($value:expr)? ) => {
//            crate::utils::test::jit_enum!(1, crate::utils::test::jit_struct!(
            crate::utils::test::jit_enum!(1, crate::utils::test::jit_struct!(
//                crate::utils::test::jit_struct!(),
                crate::utils::test::jit_struct!(),
//                [$($value), *].into()
                [$($value), *].into()
//            ))
            ))
//        };
        };
//    }
    }
//    pub(crate) use jit_dict;
    pub(crate) use jit_dict;
//    pub(crate) use jit_enum;
    pub(crate) use jit_enum;
//    pub(crate) use jit_panic;
    pub(crate) use jit_panic;
//    pub(crate) use jit_struct;
    pub(crate) use jit_struct;
//

//    pub(crate) fn load_cairo_str(program_str: &str) -> (String, Program) {
    pub(crate) fn load_cairo_str(program_str: &str) -> (String, Program) {
//        compile_program(program_str, RootDatabase::default())
        compile_program(program_str, RootDatabase::default())
//    }
    }
//

//    pub(crate) fn load_starknet_str(program_str: &str) -> (String, Program) {
    pub(crate) fn load_starknet_str(program_str: &str) -> (String, Program) {
//        compile_program(
        compile_program(
//            program_str,
            program_str,
//            RootDatabase::builder()
            RootDatabase::builder()
//                .with_plugin_suite(starknet_plugin_suite())
                .with_plugin_suite(starknet_plugin_suite())
//                .build()
                .build()
//                .unwrap(),
                .unwrap(),
//        )
        )
//    }
    }
//

//    pub(crate) fn compile_program(program_str: &str, mut db: RootDatabase) -> (String, Program) {
    pub(crate) fn compile_program(program_str: &str, mut db: RootDatabase) -> (String, Program) {
//        let mut program_file = tempfile::Builder::new()
        let mut program_file = tempfile::Builder::new()
//            .prefix("test_")
            .prefix("test_")
//            .suffix(".cairo")
            .suffix(".cairo")
//            .tempfile()
            .tempfile()
//            .unwrap();
            .unwrap();
//        fs::write(&mut program_file, program_str).unwrap();
        fs::write(&mut program_file, program_str).unwrap();
//

//        init_dev_corelib(
        init_dev_corelib(
//            &mut db,
            &mut db,
//            Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
            Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
//        );
        );
//        let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
        let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
//        let program = compile_prepared_db(
        let program = compile_prepared_db(
//            &mut db,
            &mut db,
//            main_crate_ids,
            main_crate_ids,
//            CompilerConfig {
            CompilerConfig {
//                diagnostics_reporter: DiagnosticsReporter::stderr(),
                diagnostics_reporter: DiagnosticsReporter::stderr(),
//                replace_ids: true,
                replace_ids: true,
//                ..Default::default()
                ..Default::default()
//            },
            },
//        )
        )
//        .unwrap();
        .unwrap();
//

//        let module_name = program_file.path().with_extension("");
        let module_name = program_file.path().with_extension("");
//        let module_name = module_name.file_name().unwrap().to_str().unwrap();
        let module_name = module_name.file_name().unwrap().to_str().unwrap();
//        (module_name.to_string(), program)
        (module_name.to_string(), program)
//    }
    }
//

//    pub fn run_program(
    pub fn run_program(
//        program: &(String, Program),
        program: &(String, Program),
//        entry_point: &str,
        entry_point: &str,
//        args: &[JitValue],
        args: &[JitValue],
//    ) -> ExecutionResult {
    ) -> ExecutionResult {
//        let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
        let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
//        let program = &program.1;
        let program = &program.1;
//

//        let entry_point_id = &program
        let entry_point_id = &program
//            .funcs
            .funcs
//            .iter()
            .iter()
//            .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
            .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
//            .expect("Test program entry point not found.")
            .expect("Test program entry point not found.")
//            .id;
            .id;
//

//        let context = NativeContext::new();
        let context = NativeContext::new();
//

//        let module = context
        let module = context
//            .compile(program, None)
            .compile(program, None)
//            .expect("Could not compile test program to MLIR.");
            .expect("Could not compile test program to MLIR.");
//

//        // FIXME: There are some bugs with non-zero LLVM optimization levels.
        // FIXME: There are some bugs with non-zero LLVM optimization levels.
//        let executor = JitNativeExecutor::from_native_module(module, OptLevel::None);
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::None);
//        executor
        executor
//            .invoke_dynamic_with_syscall_handler(
            .invoke_dynamic_with_syscall_handler(
//                entry_point_id,
                entry_point_id,
//                args,
                args,
//                Some(u128::MAX),
                Some(u128::MAX),
//                TestSyscallHandler,
                TestSyscallHandler,
//            )
            )
//            .unwrap()
            .unwrap()
//    }
    }
//

//    #[track_caller]
    #[track_caller]
//    pub fn run_program_assert_output(
    pub fn run_program_assert_output(
//        program: &(String, Program),
        program: &(String, Program),
//        entry_point: &str,
        entry_point: &str,
//        args: &[JitValue],
        args: &[JitValue],
//        output: JitValue,
        output: JitValue,
//    ) {
    ) {
//        let result = run_program(program, entry_point, args);
        let result = run_program(program, entry_point, args);
//        assert_eq!(result.return_value, output);
        assert_eq!(result.return_value, output);
//    }
    }
//

//    // ==============================
    // ==============================
//    // == TESTS: get_integer_layout
    // == TESTS: get_integer_layout
//    // ==============================
    // ==============================
//    /// Ensures that the host's `u8` is compatible with its compiled counterpart.
    /// Ensures that the host's `u8` is compatible with its compiled counterpart.
//    #[test]
    #[test]
//    fn test_alignment_compatibility_u8() {
    fn test_alignment_compatibility_u8() {
//        assert_eq!(get_integer_layout(8).align(), 1);
        assert_eq!(get_integer_layout(8).align(), 1);
//    }
    }
//

//    /// Ensures that the host's `u16` is compatible with its compiled counterpart.
    /// Ensures that the host's `u16` is compatible with its compiled counterpart.
//    #[test]
    #[test]
//    fn test_alignment_compatibility_u16() {
    fn test_alignment_compatibility_u16() {
//        assert_eq!(get_integer_layout(16).align(), 2);
        assert_eq!(get_integer_layout(16).align(), 2);
//    }
    }
//

//    /// Ensures that the host's `u32` is compatible with its compiled counterpart.
    /// Ensures that the host's `u32` is compatible with its compiled counterpart.
//    #[test]
    #[test]
//    fn test_alignment_compatibility_u32() {
    fn test_alignment_compatibility_u32() {
//        assert_eq!(get_integer_layout(32).align(), 4);
        assert_eq!(get_integer_layout(32).align(), 4);
//    }
    }
//

//    /// Ensures that the host's `u64` is compatible with its compiled counterpart.
    /// Ensures that the host's `u64` is compatible with its compiled counterpart.
//    #[test]
    #[test]
//    fn test_alignment_compatibility_u64() {
    fn test_alignment_compatibility_u64() {
//        assert_eq!(get_integer_layout(64).align(), 8);
        assert_eq!(get_integer_layout(64).align(), 8);
//    }
    }
//

//    /// Ensures that the host's `u128` is compatible with its compiled counterpart.
    /// Ensures that the host's `u128` is compatible with its compiled counterpart.
//    #[test]
    #[test]
//    fn test_alignment_compatibility_u128() {
    fn test_alignment_compatibility_u128() {
//        assert_eq!(get_integer_layout(128).align(), 16);
        assert_eq!(get_integer_layout(128).align(), 16);
//    }
    }
//

//    /// Ensures that the host's `u256` is compatible with its compiled counterpart.
    /// Ensures that the host's `u256` is compatible with its compiled counterpart.
//    #[test]
    #[test]
//    fn test_alignment_compatibility_u256() {
    fn test_alignment_compatibility_u256() {
//        assert_eq!(get_integer_layout(256).align(), 16);
        assert_eq!(get_integer_layout(256).align(), 16);
//    }
    }
//

//    /// Ensures that the host's `u512` is compatible with its compiled counterpart.
    /// Ensures that the host's `u512` is compatible with its compiled counterpart.
//    #[test]
    #[test]
//    fn test_alignment_compatibility_u512() {
    fn test_alignment_compatibility_u512() {
//        assert_eq!(get_integer_layout(512).align(), 16);
        assert_eq!(get_integer_layout(512).align(), 16);
//    }
    }
//

//    /// Ensures that the host's `Felt` is compatible with its compiled counterpart.
    /// Ensures that the host's `Felt` is compatible with its compiled counterpart.
//    #[test]
    #[test]
//    fn test_alignment_compatibility_felt() {
    fn test_alignment_compatibility_felt() {
//        assert_eq!(get_integer_layout(252).align(), 16);
        assert_eq!(get_integer_layout(252).align(), 16);
//    }
    }
//

//    // ==============================
    // ==============================
//    // == TESTS: find_entry_point
    // == TESTS: find_entry_point
//    // ==============================
    // ==============================
//    #[test]
    #[test]
//    fn test_find_entry_point_with_empty_program() {
    fn test_find_entry_point_with_empty_program() {
//        let program = Program {
        let program = Program {
//            type_declarations: vec![],
            type_declarations: vec![],
//            libfunc_declarations: vec![],
            libfunc_declarations: vec![],
//            statements: vec![],
            statements: vec![],
//            funcs: vec![],
            funcs: vec![],
//        };
        };
//        let entry_point = find_entry_point(&program, "entry_point");
        let entry_point = find_entry_point(&program, "entry_point");
//        assert!(entry_point.is_none());
        assert!(entry_point.is_none());
//    }
    }
//

//    #[test]
    #[test]
//    fn test_entry_point_not_found() {
    fn test_entry_point_not_found() {
//        let program = Program {
        let program = Program {
//            type_declarations: vec![],
            type_declarations: vec![],
//            libfunc_declarations: vec![],
            libfunc_declarations: vec![],
//            statements: vec![],
            statements: vec![],
//            funcs: vec![GenFunction {
            funcs: vec![GenFunction {
//                id: FunctionId {
                id: FunctionId {
//                    id: 0,
                    id: 0,
//                    debug_name: Some("not_entry_point".into()),
                    debug_name: Some("not_entry_point".into()),
//                },
                },
//                signature: FunctionSignature {
                signature: FunctionSignature {
//                    ret_types: vec![],
                    ret_types: vec![],
//                    param_types: vec![],
                    param_types: vec![],
//                },
                },
//                params: vec![],
                params: vec![],
//                entry_point: StatementIdx(0),
                entry_point: StatementIdx(0),
//            }],
            }],
//        };
        };
//        let entry_point = find_entry_point(&program, "entry_point");
        let entry_point = find_entry_point(&program, "entry_point");
//        assert!(entry_point.is_none());
        assert!(entry_point.is_none());
//    }
    }
//

//    #[test]
    #[test]
//    fn test_entry_point_found() {
    fn test_entry_point_found() {
//        let program = Program {
        let program = Program {
//            type_declarations: vec![],
            type_declarations: vec![],
//            libfunc_declarations: vec![],
            libfunc_declarations: vec![],
//            statements: vec![],
            statements: vec![],
//            funcs: vec![GenFunction {
            funcs: vec![GenFunction {
//                id: FunctionId {
                id: FunctionId {
//                    id: 0,
                    id: 0,
//                    debug_name: Some("entry_point".into()),
                    debug_name: Some("entry_point".into()),
//                },
                },
//                signature: FunctionSignature {
                signature: FunctionSignature {
//                    ret_types: vec![],
                    ret_types: vec![],
//                    param_types: vec![],
                    param_types: vec![],
//                },
                },
//                params: vec![],
                params: vec![],
//                entry_point: StatementIdx(0),
                entry_point: StatementIdx(0),
//            }],
            }],
//        };
        };
//        let entry_point = find_entry_point(&program, "entry_point");
        let entry_point = find_entry_point(&program, "entry_point");
//        assert!(entry_point.is_some());
        assert!(entry_point.is_some());
//        assert_eq!(entry_point.unwrap().id.id, 0);
        assert_eq!(entry_point.unwrap().id.id, 0);
//    }
    }
//

//    // ====================================
    // ====================================
//    // == TESTS: find_entry_point_by_idx
    // == TESTS: find_entry_point_by_idx
//    // ====================================
    // ====================================
//    #[test]
    #[test]
//    fn test_find_entry_point_by_idx_with_empty_program() {
    fn test_find_entry_point_by_idx_with_empty_program() {
//        let program = Program {
        let program = Program {
//            type_declarations: vec![],
            type_declarations: vec![],
//            libfunc_declarations: vec![],
            libfunc_declarations: vec![],
//            statements: vec![],
            statements: vec![],
//            funcs: vec![],
            funcs: vec![],
//        };
        };
//        let entry_point = find_entry_point_by_idx(&program, 0);
        let entry_point = find_entry_point_by_idx(&program, 0);
//        assert!(entry_point.is_none());
        assert!(entry_point.is_none());
//    }
    }
//

//    #[test]
    #[test]
//    fn test_entry_point_not_found_by_id() {
    fn test_entry_point_not_found_by_id() {
//        let program = Program {
        let program = Program {
//            type_declarations: vec![],
            type_declarations: vec![],
//            libfunc_declarations: vec![],
            libfunc_declarations: vec![],
//            statements: vec![],
            statements: vec![],
//            funcs: vec![GenFunction {
            funcs: vec![GenFunction {
//                id: FunctionId {
                id: FunctionId {
//                    id: 0,
                    id: 0,
//                    debug_name: Some("some_name".into()),
                    debug_name: Some("some_name".into()),
//                },
                },
//                signature: FunctionSignature {
                signature: FunctionSignature {
//                    ret_types: vec![],
                    ret_types: vec![],
//                    param_types: vec![],
                    param_types: vec![],
//                },
                },
//                params: vec![],
                params: vec![],
//                entry_point: StatementIdx(0),
                entry_point: StatementIdx(0),
//            }],
            }],
//        };
        };
//        let entry_point = find_entry_point_by_idx(&program, 1);
        let entry_point = find_entry_point_by_idx(&program, 1);
//        assert!(entry_point.is_none());
        assert!(entry_point.is_none());
//    }
    }
//

//    #[test]
    #[test]
//    fn test_entry_point_found_by_id() {
    fn test_entry_point_found_by_id() {
//        let program = Program {
        let program = Program {
//            type_declarations: vec![],
            type_declarations: vec![],
//            libfunc_declarations: vec![],
            libfunc_declarations: vec![],
//            statements: vec![],
            statements: vec![],
//            funcs: vec![GenFunction {
            funcs: vec![GenFunction {
//                id: FunctionId {
                id: FunctionId {
//                    id: 15,
                    id: 15,
//                    debug_name: Some("some_name".into()),
                    debug_name: Some("some_name".into()),
//                },
                },
//                signature: FunctionSignature {
                signature: FunctionSignature {
//                    ret_types: vec![],
                    ret_types: vec![],
//                    param_types: vec![],
                    param_types: vec![],
//                },
                },
//                params: vec![],
                params: vec![],
//                entry_point: StatementIdx(0),
                entry_point: StatementIdx(0),
//            }],
            }],
//        };
        };
//        let entry_point = find_entry_point_by_idx(&program, 15);
        let entry_point = find_entry_point_by_idx(&program, 15);
//        assert!(entry_point.is_some());
        assert!(entry_point.is_some());
//        assert_eq!(entry_point.unwrap().id.id, 15);
        assert_eq!(entry_point.unwrap().id.id, 15);
//    }
    }
//

//    // ==============================
    // ==============================
//    // == TESTS: felt252_str
    // == TESTS: felt252_str
//    // ==============================
    // ==============================
//    #[test]
    #[test]
//    #[should_panic(expected = "value must be a digit number")]
    #[should_panic(expected = "value must be a digit number")]
//    fn test_felt252_str_invalid_input() {
    fn test_felt252_str_invalid_input() {
//        let value = "not_a_number";
        let value = "not_a_number";
//        felt252_str(value);
        felt252_str(value);
//    }
    }
//

//    #[test]
    #[test]
//    fn test_felt252_str_positive_number() {
    fn test_felt252_str_positive_number() {
//        let value = "123";
        let value = "123";
//        let result = felt252_str(value);
        let result = felt252_str(value);
//        assert_eq!(result, [123, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(result, [123, 0, 0, 0, 0, 0, 0, 0]);
//    }
    }
//

//    #[test]
    #[test]
//    fn test_felt252_str_negative_number() {
    fn test_felt252_str_negative_number() {
//        let value = "-123";
        let value = "-123";
//        let result = felt252_str(value);
        let result = felt252_str(value);
//        assert_eq!(
        assert_eq!(
//            result,
            result,
//            [
            [
//                4294967174, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 16,
                4294967174, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 16,
//                134217728
                134217728
//            ]
            ]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_felt252_str_zero() {
    fn test_felt252_str_zero() {
//        let value = "0";
        let value = "0";
//        let result = felt252_str(value);
        let result = felt252_str(value);
//        assert_eq!(result, [0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(result, [0, 0, 0, 0, 0, 0, 0, 0]);
//    }
    }
//

//    // ==============================
    // ==============================
//    // == TESTS: felt252_short_str
    // == TESTS: felt252_short_str
//    // ==============================
    // ==============================
//    #[test]
    #[test]
//    fn test_felt252_short_str_short_numeric_string() {
    fn test_felt252_short_str_short_numeric_string() {
//        let value = "12345";
        let value = "12345";
//        let result = felt252_short_str(value);
        let result = felt252_short_str(value);
//        assert_eq!(result, [842216501, 49, 0, 0, 0, 0, 0, 0]);
        assert_eq!(result, [842216501, 49, 0, 0, 0, 0, 0, 0]);
//    }
    }
//

//    #[test]
    #[test]
//    fn test_felt252_short_str_short_string_with_non_numeric_characters() {
    fn test_felt252_short_str_short_string_with_non_numeric_characters() {
//        let value = "hello";
        let value = "hello";
//        let result = felt252_short_str(value);
        let result = felt252_short_str(value);
//        assert_eq!(result, [1701604463, 104, 0, 0, 0, 0, 0, 0]);
        assert_eq!(result, [1701604463, 104, 0, 0, 0, 0, 0, 0]);
//    }
    }
//

//    #[test]
    #[test]
//    fn test_felt252_short_str_long_numeric_string() {
    fn test_felt252_short_str_long_numeric_string() {
//        let value = "1234567890123456789012345678901234567890";
        let value = "1234567890123456789012345678901234567890";
//        let result = felt252_short_str(value);
        let result = felt252_short_str(value);
//        assert_eq!(
        assert_eq!(
//            result,
            result,
//            [
            [
//                926431536, 859059510, 959459634, 892745528, 825373492, 926431536, 859059510,
                926431536, 859059510, 959459634, 892745528, 825373492, 926431536, 859059510,
//                959459634
                959459634
//            ]
            ]
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_felt252_short_str_empty_string() {
    fn test_felt252_short_str_empty_string() {
//        let value = "";
        let value = "";
//        let result = felt252_short_str(value);
        let result = felt252_short_str(value);
//        assert_eq!(result, [0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(result, [0, 0, 0, 0, 0, 0, 0, 0]);
//    }
    }
//

//    #[test]
    #[test]
//    fn test_felt252_short_str_string_with_non_ascii_characters() {
    fn test_felt252_short_str_string_with_non_ascii_characters() {
//        let value = "h€llø";
        let value = "h€llø";
//        let result = felt252_short_str(value);
        let result = felt252_short_str(value);
//        assert_eq!(result, [6843500, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(result, [6843500, 0, 0, 0, 0, 0, 0, 0]);
//    }
    }
//

//    // ==============================
    // ==============================
//    // == TESTS: debug_with
    // == TESTS: debug_with
//    // ==============================
    // ==============================
//    #[test]
    #[test]
//    fn test_debug_with_empty_closure() {
    fn test_debug_with_empty_closure() {
//        let closure = |_f: &mut Formatter| -> fmt::Result { Ok(()) };
        let closure = |_f: &mut Formatter| -> fmt::Result { Ok(()) };
//        let debug_wrapper = debug_with(closure);
        let debug_wrapper = debug_with(closure);
//        assert_eq!(format!("{:?}", debug_wrapper), "");
        assert_eq!(format!("{:?}", debug_wrapper), "");
//    }
    }
//

//    #[test]
    #[test]
//    #[should_panic]
    #[should_panic]
//    fn test_debug_with_error_closure() {
    fn test_debug_with_error_closure() {
//        let closure = |_f: &mut Formatter| -> Result<(), fmt::Error> { Err(fmt::Error) };
        let closure = |_f: &mut Formatter| -> Result<(), fmt::Error> { Err(fmt::Error) };
//        let debug_wrapper = debug_with(closure);
        let debug_wrapper = debug_with(closure);
//        let _ = format!("{:?}", debug_wrapper);
        let _ = format!("{:?}", debug_wrapper);
//    }
    }
//

//    #[test]
    #[test]
//    fn test_debug_with_simple_closure() {
    fn test_debug_with_simple_closure() {
//        let closure = |f: &mut fmt::Formatter| write!(f, "Hello, world!");
        let closure = |f: &mut fmt::Formatter| write!(f, "Hello, world!");
//        let debug_wrapper = debug_with(closure);
        let debug_wrapper = debug_with(closure);
//        assert_eq!(format!("{:?}", debug_wrapper), "Hello, world!");
        assert_eq!(format!("{:?}", debug_wrapper), "Hello, world!");
//    }
    }
//

//    #[test]
    #[test]
//    fn test_debug_with_complex_closure() {
    fn test_debug_with_complex_closure() {
//        let closure = |f: &mut fmt::Formatter| write!(f, "Name: William, Age: {}", 28);
        let closure = |f: &mut fmt::Formatter| write!(f, "Name: William, Age: {}", 28);
//        let debug_wrapper = debug_with(closure);
        let debug_wrapper = debug_with(closure);
//        assert_eq!(format!("{:?}", debug_wrapper), "Name: William, Age: 28");
        assert_eq!(format!("{:?}", debug_wrapper), "Name: William, Age: 28");
//    }
    }
//

//    #[test]
    #[test]
//    fn test_generate_function_name_debug_name() {
    fn test_generate_function_name_debug_name() {
//        let function_id = FunctionId {
        let function_id = FunctionId {
//            id: 123,
            id: 123,
//            debug_name: Some("function_name".into()),
            debug_name: Some("function_name".into()),
//        };
        };
//

//        assert_eq!(generate_function_name(&function_id), "function_name(f123)");
        assert_eq!(generate_function_name(&function_id), "function_name(f123)");
//    }
    }
//

//    #[test]
    #[test]
//    fn test_generate_function_name_without_debug_name() {
    fn test_generate_function_name_without_debug_name() {
//        let function_id = FunctionId {
        let function_id = FunctionId {
//            id: 123,
            id: 123,
//            debug_name: None,
            debug_name: None,
//        };
        };
//

//        assert_eq!(generate_function_name(&function_id), "f123");
        assert_eq!(generate_function_name(&function_id), "f123");
//    }
    }
//

//    #[test]
    #[test]
//    fn test_cairo_to_sierra_path() {
    fn test_cairo_to_sierra_path() {
//        // Define the path to the cairo program.
        // Define the path to the cairo program.
//        let program_path = Path::new("programs/examples/hello.cairo");
        let program_path = Path::new("programs/examples/hello.cairo");
//        // Compile the cairo program to sierra.
        // Compile the cairo program to sierra.
//        let sierra_program = cairo_to_sierra(program_path);
        let sierra_program = cairo_to_sierra(program_path);
//

//        // Define the entry point function for comparison.
        // Define the entry point function for comparison.
//        let entry_point = "hello::hello::greet";
        let entry_point = "hello::hello::greet";
//        // Find the function ID of the entry point function in the sierra program.
        // Find the function ID of the entry point function in the sierra program.
//        let entry_point_id = find_function_id(&sierra_program, entry_point);
        let entry_point_id = find_function_id(&sierra_program, entry_point);
//

//        // Assert that the debug name of the entry point function matches the expected value.
        // Assert that the debug name of the entry point function matches the expected value.
//        assert_eq!(
        assert_eq!(
//            entry_point_id.debug_name,
            entry_point_id.debug_name,
//            Some("hello::hello::greet".into())
            Some("hello::hello::greet".into())
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_cairo_to_sierra_source() {
    fn test_cairo_to_sierra_source() {
//        // Define the content of the cairo program as a string.
        // Define the content of the cairo program as a string.
//        let content = "type u8 = u8;";
        let content = "type u8 = u8;";
//

//        // Create a named temporary file and write the content to it.
        // Create a named temporary file and write the content to it.
//        let mut file = tempfile::NamedTempFile::new().unwrap();
        let mut file = tempfile::NamedTempFile::new().unwrap();
//        file.write_all(content.as_bytes()).unwrap();
        file.write_all(content.as_bytes()).unwrap();
//        // Get the path of the temporary file.
        // Get the path of the temporary file.
//        let file_path = file.path().to_path_buf();
        let file_path = file.path().to_path_buf();
//

//        // Compile the cairo program to sierra using the path of the temporary file.
        // Compile the cairo program to sierra using the path of the temporary file.
//        let sierra_program = cairo_to_sierra(&file_path);
        let sierra_program = cairo_to_sierra(&file_path);
//

//        // Assert that the sierra program has no library function declarations, statements, or functions.
        // Assert that the sierra program has no library function declarations, statements, or functions.
//        assert!(sierra_program.libfunc_declarations.is_empty());
        assert!(sierra_program.libfunc_declarations.is_empty());
//        assert!(sierra_program.statements.is_empty());
        assert!(sierra_program.statements.is_empty());
//        assert!(sierra_program.funcs.is_empty());
        assert!(sierra_program.funcs.is_empty());
//

//        // Assert that the debug name of the first type declaration matches the expected value.
        // Assert that the debug name of the first type declaration matches the expected value.
//        assert_eq!(sierra_program.type_declarations.len(), 1);
        assert_eq!(sierra_program.type_declarations.len(), 1);
//        assert_eq!(
        assert_eq!(
//            sierra_program.type_declarations[0].id.debug_name,
            sierra_program.type_declarations[0].id.debug_name,
//            Some("u8".into())
            Some("u8".into())
//        );
        );
//    }
    }
//

//    #[test]
    #[test]
//    fn test_cairo_to_sierra_with_debug_info() {
    fn test_cairo_to_sierra_with_debug_info() {
//        // Define the path to the cairo program.
        // Define the path to the cairo program.
//        let program_path = Path::new("programs/examples/hello.cairo");
        let program_path = Path::new("programs/examples/hello.cairo");
//        // Create a new context.
        // Create a new context.
//        let context = Context::new();
        let context = Context::new();
//        // Compile the cairo program to sierra, including debug information.
        // Compile the cairo program to sierra, including debug information.
//        let sierra_program = cairo_to_sierra_with_debug_info(&context, program_path).unwrap();
        let sierra_program = cairo_to_sierra_with_debug_info(&context, program_path).unwrap();
//

//        // Define the name of the entry point function for comparison.
        // Define the name of the entry point function for comparison.
//        let entry_point = "hello::hello::greet";
        let entry_point = "hello::hello::greet";
//        // Find the function ID of the entry point function in the sierra program.
        // Find the function ID of the entry point function in the sierra program.
//        let entry_point_id = find_function_id(&sierra_program.0, entry_point);
        let entry_point_id = find_function_id(&sierra_program.0, entry_point);
//

//        // Assert that the debug name of the entry point function matches the expected value.
        // Assert that the debug name of the entry point function matches the expected value.
//        assert_eq!(
        assert_eq!(
//            entry_point_id.debug_name,
            entry_point_id.debug_name,
//            Some("hello::hello::greet".into())
            Some("hello::hello::greet".into())
//        );
        );
//

//        // Check if the sierra program contains a function with the specified debug name for the entry point function.
        // Check if the sierra program contains a function with the specified debug name for the entry point function.
//        assert!(sierra_program
        assert!(sierra_program
//            .1
            .1
//            .funcs
            .funcs
//            .keys()
            .keys()
//            .any(|func| func.debug_name == Some("hello::hello::greet".into())));
            .any(|func| func.debug_name == Some("hello::hello::greet".into())));
//    }
    }
//

//    #[derive(Debug, Clone)]
    #[derive(Debug, Clone)]
//    pub struct TestSyscallHandler;
    pub struct TestSyscallHandler;
//

//    impl StarknetSyscallHandler for TestSyscallHandler {
    impl StarknetSyscallHandler for TestSyscallHandler {
//        fn get_block_hash(&mut self, block_number: u64, _gas: &mut u128) -> SyscallResult<Felt> {
        fn get_block_hash(&mut self, block_number: u64, _gas: &mut u128) -> SyscallResult<Felt> {
//            Ok(Felt::from(block_number))
            Ok(Felt::from(block_number))
//        }
        }
//

//        fn get_execution_info(
        fn get_execution_info(
//            &mut self,
            &mut self,
//            _gas: &mut u128,
            _gas: &mut u128,
//        ) -> SyscallResult<crate::starknet::ExecutionInfo> {
        ) -> SyscallResult<crate::starknet::ExecutionInfo> {
//            Ok(ExecutionInfo {
            Ok(ExecutionInfo {
//                block_info: BlockInfo {
                block_info: BlockInfo {
//                    block_number: 1234,
                    block_number: 1234,
//                    block_timestamp: 2345,
                    block_timestamp: 2345,
//                    sequencer_address: 3456.into(),
                    sequencer_address: 3456.into(),
//                },
                },
//                tx_info: TxInfo {
                tx_info: TxInfo {
//                    version: 4567.into(),
                    version: 4567.into(),
//                    account_contract_address: 5678.into(),
                    account_contract_address: 5678.into(),
//                    max_fee: 6789,
                    max_fee: 6789,
//                    signature: vec![1248.into(), 2486.into()],
                    signature: vec![1248.into(), 2486.into()],
//                    transaction_hash: 9876.into(),
                    transaction_hash: 9876.into(),
//                    chain_id: 8765.into(),
                    chain_id: 8765.into(),
//                    nonce: 7654.into(),
                    nonce: 7654.into(),
//                },
                },
//                caller_address: 6543.into(),
                caller_address: 6543.into(),
//                contract_address: 5432.into(),
                contract_address: 5432.into(),
//                entry_point_selector: 4321.into(),
                entry_point_selector: 4321.into(),
//            })
            })
//        }
        }
//

//        fn get_execution_info_v2(
        fn get_execution_info_v2(
//            &mut self,
            &mut self,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<crate::starknet::ExecutionInfoV2> {
        ) -> SyscallResult<crate::starknet::ExecutionInfoV2> {
//            Ok(ExecutionInfoV2 {
            Ok(ExecutionInfoV2 {
//                block_info: BlockInfo {
                block_info: BlockInfo {
//                    block_number: 1234,
                    block_number: 1234,
//                    block_timestamp: 2345,
                    block_timestamp: 2345,
//                    sequencer_address: 3456.into(),
                    sequencer_address: 3456.into(),
//                },
                },
//                tx_info: TxV2Info {
                tx_info: TxV2Info {
//                    version: 1.into(),
                    version: 1.into(),
//                    account_contract_address: 1.into(),
                    account_contract_address: 1.into(),
//                    max_fee: 0,
                    max_fee: 0,
//                    signature: vec![1.into()],
                    signature: vec![1.into()],
//                    transaction_hash: 1.into(),
                    transaction_hash: 1.into(),
//                    chain_id: 1.into(),
                    chain_id: 1.into(),
//                    nonce: 1.into(),
                    nonce: 1.into(),
//                    tip: 1,
                    tip: 1,
//                    paymaster_data: vec![1.into()],
                    paymaster_data: vec![1.into()],
//                    nonce_data_availability_mode: 0,
                    nonce_data_availability_mode: 0,
//                    fee_data_availability_mode: 0,
                    fee_data_availability_mode: 0,
//                    account_deployment_data: vec![1.into()],
                    account_deployment_data: vec![1.into()],
//                    resource_bounds: vec![ResourceBounds {
                    resource_bounds: vec![ResourceBounds {
//                        resource: 2.into(),
                        resource: 2.into(),
//                        max_amount: 10,
                        max_amount: 10,
//                        max_price_per_unit: 20,
                        max_price_per_unit: 20,
//                    }],
                    }],
//                },
                },
//                caller_address: 6543.into(),
                caller_address: 6543.into(),
//                contract_address: 5432.into(),
                contract_address: 5432.into(),
//                entry_point_selector: 4321.into(),
                entry_point_selector: 4321.into(),
//            })
            })
//        }
        }
//

//        fn deploy(
        fn deploy(
//            &mut self,
            &mut self,
//            class_hash: Felt,
            class_hash: Felt,
//            contract_address_salt: Felt,
            contract_address_salt: Felt,
//            calldata: &[Felt],
            calldata: &[Felt],
//            _deploy_from_zero: bool,
            _deploy_from_zero: bool,
//            _gas: &mut u128,
            _gas: &mut u128,
//        ) -> SyscallResult<(Felt, Vec<Felt>)> {
        ) -> SyscallResult<(Felt, Vec<Felt>)> {
//            Ok((
            Ok((
//                class_hash + contract_address_salt,
                class_hash + contract_address_salt,
//                calldata.iter().map(|x| x + Felt::ONE).collect(),
                calldata.iter().map(|x| x + Felt::ONE).collect(),
//            ))
            ))
//        }
        }
//

//        fn replace_class(&mut self, _class_hash: Felt, _gas: &mut u128) -> SyscallResult<()> {
        fn replace_class(&mut self, _class_hash: Felt, _gas: &mut u128) -> SyscallResult<()> {
//            Ok(())
            Ok(())
//        }
        }
//

//        fn library_call(
        fn library_call(
//            &mut self,
            &mut self,
//            _class_hash: Felt,
            _class_hash: Felt,
//            _function_selector: Felt,
            _function_selector: Felt,
//            calldata: &[Felt],
            calldata: &[Felt],
//            _gas: &mut u128,
            _gas: &mut u128,
//        ) -> SyscallResult<Vec<Felt>> {
        ) -> SyscallResult<Vec<Felt>> {
//            Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
            Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
//        }
        }
//

//        fn call_contract(
        fn call_contract(
//            &mut self,
            &mut self,
//            _address: Felt,
            _address: Felt,
//            _entry_point_selector: Felt,
            _entry_point_selector: Felt,
//            calldata: &[Felt],
            calldata: &[Felt],
//            _gas: &mut u128,
            _gas: &mut u128,
//        ) -> SyscallResult<Vec<Felt>> {
        ) -> SyscallResult<Vec<Felt>> {
//            Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
            Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
//        }
        }
//

//        fn storage_read(
        fn storage_read(
//            &mut self,
            &mut self,
//            _address_domain: u32,
            _address_domain: u32,
//            address: Felt,
            address: Felt,
//            _gas: &mut u128,
            _gas: &mut u128,
//        ) -> SyscallResult<Felt> {
        ) -> SyscallResult<Felt> {
//            Ok(address * Felt::from(3))
            Ok(address * Felt::from(3))
//        }
        }
//

//        fn storage_write(
        fn storage_write(
//            &mut self,
            &mut self,
//            _address_domain: u32,
            _address_domain: u32,
//            _address: Felt,
            _address: Felt,
//            _value: Felt,
            _value: Felt,
//            _gas: &mut u128,
            _gas: &mut u128,
//        ) -> SyscallResult<()> {
        ) -> SyscallResult<()> {
//            Ok(())
            Ok(())
//        }
        }
//

//        fn emit_event(
        fn emit_event(
//            &mut self,
            &mut self,
//            _keys: &[Felt],
            _keys: &[Felt],
//            _data: &[Felt],
            _data: &[Felt],
//            _gas: &mut u128,
            _gas: &mut u128,
//        ) -> SyscallResult<()> {
        ) -> SyscallResult<()> {
//            Ok(())
            Ok(())
//        }
        }
//

//        fn send_message_to_l1(
        fn send_message_to_l1(
//            &mut self,
            &mut self,
//            _to_address: Felt,
            _to_address: Felt,
//            _payload: &[Felt],
            _payload: &[Felt],
//            _gas: &mut u128,
            _gas: &mut u128,
//        ) -> SyscallResult<()> {
        ) -> SyscallResult<()> {
//            Ok(())
            Ok(())
//        }
        }
//

//        fn keccak(
        fn keccak(
//            &mut self,
            &mut self,
//            _input: &[u64],
            _input: &[u64],
//            gas: &mut u128,
            gas: &mut u128,
//        ) -> SyscallResult<crate::starknet::U256> {
        ) -> SyscallResult<crate::starknet::U256> {
//            *gas -= 1000;
            *gas -= 1000;
//            Ok(U256 {
            Ok(U256 {
//                hi: 0,
                hi: 0,
//                lo: 1234567890,
                lo: 1234567890,
//            })
            })
//        }
        }
//

//        // Implementing the secp256 syscalls for testing doesn't make sense. They're already
        // Implementing the secp256 syscalls for testing doesn't make sense. They're already
//        // properly tested in integration tests.
        // properly tested in integration tests.
//

//        fn secp256k1_new(
        fn secp256k1_new(
//            &mut self,
            &mut self,
//            _x: U256,
            _x: U256,
//            _y: U256,
            _y: U256,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<Option<crate::starknet::Secp256k1Point>> {
        ) -> SyscallResult<Option<crate::starknet::Secp256k1Point>> {
//            unimplemented!()
            unimplemented!()
//        }
        }
//

//        fn secp256k1_add(
        fn secp256k1_add(
//            &mut self,
            &mut self,
//            _p0: crate::starknet::Secp256k1Point,
            _p0: crate::starknet::Secp256k1Point,
//            _p1: crate::starknet::Secp256k1Point,
            _p1: crate::starknet::Secp256k1Point,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<crate::starknet::Secp256k1Point> {
        ) -> SyscallResult<crate::starknet::Secp256k1Point> {
//            unimplemented!()
            unimplemented!()
//        }
        }
//

//        fn secp256k1_mul(
        fn secp256k1_mul(
//            &mut self,
            &mut self,
//            _p: crate::starknet::Secp256k1Point,
            _p: crate::starknet::Secp256k1Point,
//            _m: U256,
            _m: U256,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<crate::starknet::Secp256k1Point> {
        ) -> SyscallResult<crate::starknet::Secp256k1Point> {
//            unimplemented!()
            unimplemented!()
//        }
        }
//

//        fn secp256k1_get_point_from_x(
        fn secp256k1_get_point_from_x(
//            &mut self,
            &mut self,
//            _x: U256,
            _x: U256,
//            _y_parity: bool,
            _y_parity: bool,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<Option<crate::starknet::Secp256k1Point>> {
        ) -> SyscallResult<Option<crate::starknet::Secp256k1Point>> {
//            unimplemented!()
            unimplemented!()
//        }
        }
//

//        fn secp256k1_get_xy(
        fn secp256k1_get_xy(
//            &mut self,
            &mut self,
//            _p: crate::starknet::Secp256k1Point,
            _p: crate::starknet::Secp256k1Point,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<(U256, U256)> {
        ) -> SyscallResult<(U256, U256)> {
//            unimplemented!()
            unimplemented!()
//        }
        }
//

//        fn secp256r1_new(
        fn secp256r1_new(
//            &mut self,
            &mut self,
//            _x: U256,
            _x: U256,
//            _y: U256,
            _y: U256,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<Option<crate::starknet::Secp256r1Point>> {
        ) -> SyscallResult<Option<crate::starknet::Secp256r1Point>> {
//            unimplemented!()
            unimplemented!()
//        }
        }
//

//        fn secp256r1_add(
        fn secp256r1_add(
//            &mut self,
            &mut self,
//            _p0: crate::starknet::Secp256r1Point,
            _p0: crate::starknet::Secp256r1Point,
//            _p1: crate::starknet::Secp256r1Point,
            _p1: crate::starknet::Secp256r1Point,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<crate::starknet::Secp256r1Point> {
        ) -> SyscallResult<crate::starknet::Secp256r1Point> {
//            unimplemented!()
            unimplemented!()
//        }
        }
//

//        fn secp256r1_mul(
        fn secp256r1_mul(
//            &mut self,
            &mut self,
//            _p: crate::starknet::Secp256r1Point,
            _p: crate::starknet::Secp256r1Point,
//            _m: U256,
            _m: U256,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<crate::starknet::Secp256r1Point> {
        ) -> SyscallResult<crate::starknet::Secp256r1Point> {
//            unimplemented!()
            unimplemented!()
//        }
        }
//

//        fn secp256r1_get_point_from_x(
        fn secp256r1_get_point_from_x(
//            &mut self,
            &mut self,
//            _x: U256,
            _x: U256,
//            _y_parity: bool,
            _y_parity: bool,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<Option<crate::starknet::Secp256r1Point>> {
        ) -> SyscallResult<Option<crate::starknet::Secp256r1Point>> {
//            unimplemented!()
            unimplemented!()
//        }
        }
//

//        fn secp256r1_get_xy(
        fn secp256r1_get_xy(
//            &mut self,
            &mut self,
//            _p: crate::starknet::Secp256r1Point,
            _p: crate::starknet::Secp256r1Point,
//            _remaining_gas: &mut u128,
            _remaining_gas: &mut u128,
//        ) -> SyscallResult<(U256, U256)> {
        ) -> SyscallResult<(U256, U256)> {
//            unimplemented!()
            unimplemented!()
//        }
        }
//    }
    }
//}
}
