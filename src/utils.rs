//! # Various utilities

use crate::{
    debug_info::{DebugInfo, DebugLocations},
    metadata::MetadataStorage,
    types::{felt252::PRIME, TypeBuilder},
    OptLevel,
};
use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
    project::setup_project, CompilerConfig,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        utils::Range,
    },
    ids::{ConcreteTypeId, FunctionId},
    program::{GenFunction, Program, StatementIdx},
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Module, Type},
    pass::{self, PassManager},
    Context, Error, ExecutionEngine,
};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::One;
use std::{
    alloc::Layout,
    borrow::Cow,
    fmt::{self, Display},
    ops::Neg,
    path::Path,
    ptr::NonNull,
    sync::Arc,
};
use thiserror::Error;

#[cfg(target_os = "macos")]
pub const SHARED_LIBRARY_EXT: &str = "dylib";
#[cfg(target_os = "linux")]
pub const SHARED_LIBRARY_EXT: &str = "so";

/// Generate a function name.
///
/// If the program includes function identifiers, return those. Otherwise return `f` followed by the
/// identifier number.
pub fn generate_function_name(function_id: &FunctionId) -> Cow<str> {
    // Generic functions can omit their type in the debug_name, leading to multiple functions
    // having the same name, we solve this by adding the id number even if the function has a debug_name
    if let Some(name) = function_id.debug_name.as_deref() {
        Cow::Owned(format!("{}(f{})", name, function_id.id))
    } else {
        Cow::Owned(format!("f{}", function_id.id))
    }
}

/// Return the layout for an integer of arbitrary width.
///
/// This assumes the platform's maximum (effective) alignment is 16 bytes, and that every integer
/// with a size in bytes of a power of two has the same alignment as its size.
pub fn get_integer_layout(width: u32) -> Layout {
    if width == 0 {
        Layout::new::<()>()
    } else if width <= 8 {
        Layout::new::<u8>()
    } else if width <= 16 {
        Layout::new::<u16>()
    } else if width <= 32 {
        Layout::new::<u32>()
    } else if width <= 64 {
        Layout::new::<u64>()
    } else if width <= 128 {
        #[cfg(not(target_arch = "x86_64"))]
        {
            Layout::new::<u128>()
        }
        #[cfg(target_arch = "x86_64")]
        {
            Layout::new::<u128>().align_to(16).unwrap()
        }
    } else {
        let width = (width as usize).next_multiple_of(8).next_power_of_two();
        Layout::from_size_align(width >> 3, (width >> 3).min(16)).unwrap()
    }
}

/// Compile a cairo program found at the given path to sierra.
pub fn cairo_to_sierra(program: &Path) -> Arc<Program> {
    if program
        .extension()
        .map(|x| {
            x.to_ascii_lowercase()
                .to_string_lossy()
                .eq_ignore_ascii_case("cairo")
        })
        .unwrap_or(false)
    {
        cairo_lang_compiler::compile_cairo_project_at_path(
            program,
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap()
        .into()
    } else {
        let source = std::fs::read_to_string(program).unwrap();
        cairo_lang_sierra::ProgramParser::new()
            .parse(&source)
            .unwrap()
            .into()
    }
}

pub fn cairo_to_sierra_with_debug_info<'ctx>(
    context: &'ctx Context,
    program: &Path,
) -> Result<(Program, DebugLocations<'ctx>), crate::error::Error> {
    let mut db = RootDatabase::builder().detect_corelib().build().unwrap();
    let main_crate_ids = setup_project(&mut db, program).unwrap();
    let sierra_program_with_dbg = compile_prepared_db(
        &mut db,
        main_crate_ids,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let debug_locations = {
        let debug_info = DebugInfo::extract(&db, &sierra_program_with_dbg.program)
            .map_err(|_| {
                let mut buffer = String::new();
                assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
                buffer
            })
            .unwrap();

        DebugLocations::extract(context, &db, &debug_info)
    };

    Ok((sierra_program_with_dbg.program, debug_locations))
}

/// Returns the given entry point if present.
pub fn find_entry_point<'a>(
    program: &'a Program,
    entry_point: &str,
) -> Option<&'a GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(entry_point))
}

/// Returns the given entry point if present.
pub fn find_entry_point_by_idx(
    program: &Program,
    entry_point_idx: usize,
) -> Option<&GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.id == entry_point_idx as u64)
}

/// Given a string representing a function name, searches in the program for the id corresponding to said function, and returns a reference to it.
#[track_caller]
pub fn find_function_id<'a>(program: &'a Program, function_name: &str) -> &'a FunctionId {
    &program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(function_name))
        .unwrap()
        .id
}

/// Parse a numeric string into felt, wrapping negatives around the prime modulo.
pub fn felt252_str(value: &str) -> [u32; 8] {
    let value = value
        .parse::<BigInt>()
        .expect("value must be a digit number");
    let value = match value.sign() {
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
    };

    let mut u32_digits = value.to_u32_digits();
    u32_digits.resize(8, 0);
    u32_digits.try_into().unwrap()
}

/// Parse any type that can be a bigint to a felt that can be used in the cairo-native input.
pub fn felt252_bigint(value: impl Into<BigInt>) -> [u32; 8] {
    let value: BigInt = value.into();
    let value = match value.sign() {
        Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
        _ => value.to_biguint().unwrap(),
    };

    let mut u32_digits = value.to_u32_digits();
    u32_digits.resize(8, 0);
    u32_digits.try_into().unwrap()
}

/// Parse a short string into a felt that can be used in the cairo-native input.
pub fn felt252_short_str(value: &str) -> [u32; 8] {
    let values: Vec<_> = value
        .chars()
        .filter(|&c| c.is_ascii())
        .map(|c| c as u8)
        .collect();

    let mut digits = BigUint::from_bytes_be(&values).to_u32_digits();
    digits.resize(8, 0);
    digits.try_into().unwrap()
}

/// Creates the execution engine, with all symbols registered.
pub fn create_engine(
    module: &Module,
    _metadata: &MetadataStorage,
    opt_level: OptLevel,
) -> ExecutionEngine {
    // Create the JIT engine.
    let engine = ExecutionEngine::new(module, opt_level.into(), &[], false);

    #[cfg(feature = "with-runtime")]
    register_runtime_symbols(&engine);

    #[cfg(feature = "with-debug-utils")]
    _metadata
        .get::<crate::metadata::debug_utils::DebugUtils>()
        .unwrap()
        .register_impls(&engine);

    engine
}

pub fn run_pass_manager(context: &Context, module: &mut Module) -> Result<(), Error> {
    let pass_manager = PassManager::new(context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_index_to_llvm());
    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager.run(module)
}

#[cfg(feature = "with-runtime")]
pub fn register_runtime_symbols(engine: &ExecutionEngine) {
    use cairo_native_runtime::FeltDict;

    unsafe {
        engine.register_symbol(
            "cairo_native__libfunc__debug__print",
            cairo_native_runtime::cairo_native__libfunc__debug__print
                as *const fn(i32, *const [u8; 32], usize) -> i32 as *mut (),
        );

        engine.register_symbol(
            "cairo_native__libfunc__pedersen",
            cairo_native_runtime::cairo_native__libfunc__pedersen
                as *const fn(*mut u8, *mut u8, *mut u8) -> () as *mut (),
        );

        engine.register_symbol(
            "cairo_native__libfunc__hades_permutation",
            cairo_native_runtime::cairo_native__libfunc__hades_permutation
                as *const fn(*mut u8, *mut u8, *mut u8) -> () as *mut (),
        );

        engine.register_symbol(
            "cairo_native__libfunc__ec__ec_point_from_x_nz",
            cairo_native_runtime::cairo_native__libfunc__ec__ec_point_from_x_nz
                as *const fn(*mut [[u8; 32]; 2]) -> bool as *mut (),
        );

        engine.register_symbol(
            "cairo_native__libfunc__ec__ec_state_add",
            cairo_native_runtime::cairo_native__libfunc__ec__ec_state_add
                as *const fn(*mut [[u8; 32]; 4], *const [[u8; 32]; 2]) -> bool
                as *mut (),
        );

        engine.register_symbol(
            "cairo_native__libfunc__ec__ec_state_add_mul",
            cairo_native_runtime::cairo_native__libfunc__ec__ec_state_add_mul
                as *const fn(*mut [[u8; 32]; 4], *const [u8; 32], *const [[u8; 32]; 2]) -> bool
                as *mut (),
        );

        engine.register_symbol(
            "cairo_native__libfunc__ec__ec_state_try_finalize_nz",
            cairo_native_runtime::cairo_native__libfunc__ec__ec_state_try_finalize_nz
                as *const fn(*const [[u8; 32]; 2], *mut [[u8; 32]; 4]) -> bool
                as *mut (),
        );

        engine.register_symbol(
            "cairo_native__libfunc__ec__ec_point_try_new_nz",
            cairo_native_runtime::cairo_native__libfunc__ec__ec_point_try_new_nz
                as *const fn(*const [[u8; 32]; 2]) -> bool as *mut (),
        );

        engine.register_symbol(
            "cairo_native__alloc_dict",
            cairo_native_runtime::cairo_native__alloc_dict as *const fn() -> *mut FeltDict
                as *mut (),
        );

        engine.register_symbol(
            "cairo_native__dict_free",
            cairo_native_runtime::cairo_native__dict_free as *const fn(*mut FeltDict) -> ()
                as *mut (),
        );

        engine.register_symbol(
            "cairo_native__dict_values",
            cairo_native_runtime::cairo_native__dict_values
                as *const fn(
                    *mut FeltDict,
                    *mut u64,
                ) -> *mut ([u8; 32], std::ptr::NonNull<libc::c_void>) as *mut (),
        );

        engine.register_symbol(
            "cairo_native__dict_get",
            cairo_native_runtime::cairo_native__dict_get
                as *const fn(*mut FeltDict, &[u8; 32]) -> *mut std::ffi::c_void
                as *mut (),
        );

        engine.register_symbol(
            "cairo_native__dict_insert",
            cairo_native_runtime::cairo_native__dict_insert
                as *const fn(
                    *mut FeltDict,
                    &[u8; 32],
                    NonNull<std::ffi::c_void>,
                    usize,
                ) -> *mut std::ffi::c_void as *mut (),
        );

        engine.register_symbol(
            "cairo_native__dict_gas_refund",
            cairo_native_runtime::cairo_native__dict_gas_refund as *const fn(*const FeltDict) -> u64
                as *mut (),
        );

        #[cfg(feature = "with-cheatcode")]
        {
            engine.register_symbol(
                "cairo_native__vtable_cheatcode",
                crate::starknet::cairo_native__vtable_cheatcode as *mut (),
            );
        }
    }
}

/// Return a type that calls a closure when formatted using [Debug](std::fmt::Debug).
pub fn debug_with<F>(fmt: F) -> impl fmt::Debug
where
    F: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    struct FmtWrapper<F>(F)
    where
        F: Fn(&mut fmt::Formatter) -> fmt::Result;

    impl<F> fmt::Debug for FmtWrapper<F>
    where
        F: Fn(&mut fmt::Formatter) -> fmt::Result,
    {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            self.0(f)
        }
    }

    FmtWrapper(fmt)
}

// POLYFILLS of nightly features

#[inline]
pub const fn next_multiple_of_usize(lhs: usize, rhs: usize) -> usize {
    match lhs % rhs {
        0 => lhs,
        r => lhs + (rhs - r),
    }
}

#[inline]
pub const fn next_multiple_of_u32(lhs: u32, rhs: u32) -> u32 {
    match lhs % rhs {
        0 => lhs,
        r => lhs + (rhs - r),
    }
}

/// Edit: Copied from the std lib.
///
/// Returns the amount of padding we must insert after `layout`
/// to ensure that the following address will satisfy `align`
/// (measured in bytes).
///
/// e.g., if `layout.size()` is 9, then `layout.padding_needed_for(4)`
/// returns 3, because that is the minimum number of bytes of
/// padding required to get a 4-aligned address (assuming that the
/// corresponding memory block starts at a 4-aligned address).
///
/// The return value of this function has no meaning if `align` is
/// not a power-of-two.
///
/// Note that the utility of the returned value requires `align`
/// to be less than or equal to the alignment of the starting
/// address for the whole allocated block of memory. One way to
/// satisfy this constraint is to ensure `align <= layout.align()`.
#[inline]
pub const fn padding_needed_for(layout: &Layout, align: usize) -> usize {
    let len = layout.size();

    // Rounded up value is:
    //   len_rounded_up = (len + align - 1) & !(align - 1);
    // and then we return the padding difference: `len_rounded_up - len`.
    //
    // We use modular arithmetic throughout:
    //
    // 1. align is guaranteed to be > 0, so align - 1 is always
    //    valid.
    //
    // 2. `len + align - 1` can overflow by at most `align - 1`,
    //    so the &-mask with `!(align - 1)` will ensure that in the
    //    case of overflow, `len_rounded_up` will itself be 0.
    //    Thus the returned padding, when added to `len`, yields 0,
    //    which trivially satisfies the alignment `align`.
    //
    // (Of course, attempts to allocate blocks of memory whose
    // size and padding overflow in the above manner should cause
    // the allocator to yield an error anyway.)

    let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);
    len_rounded_up.wrapping_sub(len)
}

#[derive(Clone, PartialEq, Eq, Debug, Error)]
pub struct LayoutError;

impl Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("layout error")
    }
}

/// Copied from std.
///
/// Creates a layout describing the record for `n` instances of
/// `self`, with a suitable amount of padding between each to
/// ensure that each instance is given its requested size and
/// alignment. On success, returns `(k, offs)` where `k` is the
/// layout of the array and `offs` is the distance between the start
/// of each element in the array.
///
/// On arithmetic overflow, returns `LayoutError`.
//#[unstable(feature = "alloc_layout_extra", issue = "55724")]
#[inline]
pub fn layout_repeat(layout: &Layout, n: usize) -> Result<(Layout, usize), LayoutError> {
    // This cannot overflow. Quoting from the invariant of Layout:
    // > `size`, when rounded up to the nearest multiple of `align`,
    // > must not overflow isize (i.e., the rounded value must be
    // > less than or equal to `isize::MAX`)
    let padded_size = layout.size() + padding_needed_for(layout, layout.align());
    let alloc_size = padded_size.checked_mul(n).ok_or(LayoutError)?;

    // The safe constructor is called here to enforce the isize size limit.
    let layout = Layout::from_size_align(alloc_size, layout.align()).map_err(|_| LayoutError)?;
    Ok((layout, padded_size))
}

pub trait ProgramRegistryExt {
    fn build_type<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
    ) -> Result<Type<'ctx>, super::error::Error>;

    fn build_type_with_layout<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
    ) -> Result<(Type<'ctx>, Layout), super::error::Error>;
}

impl ProgramRegistryExt for ProgramRegistry<CoreType, CoreLibfunc> {
    fn build_type<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
    ) -> Result<Type<'ctx>, super::error::Error> {
        registry
            .get_type(id)?
            .build(context, module, registry, metadata, id)
    }

    fn build_type_with_layout<'ctx>(
        &self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        metadata: &mut MetadataStorage,
        id: &ConcreteTypeId,
    ) -> Result<(Type<'ctx>, Layout), super::error::Error> {
        let concrete_type = registry.get_type(id)?;

        Ok((
            concrete_type.build(context, module, registry, metadata, id)?,
            concrete_type.layout(registry)?,
        ))
    }
}

pub trait RangeExt {
    /// Width in bits when the offset is zero (aka. the natural representation).
    fn zero_based_bit_width(&self) -> u32;
    /// Width in bits when the offset is not necessarily zero (aka. the compact representation).
    fn offset_bit_width(&self) -> u32;
}

impl RangeExt for Range {
    fn zero_based_bit_width(&self) -> u32 {
        // Formula for unsigned integers:
        //     x.bits()
        //
        // Formula for signed values:
        //   - Positive: (x.magnitude() + BigUint::one()).bits()
        //   - Negative: (x.magnitude() - BigUint::one()).bits() + 1
        //   - Zero: 0

        if self.lower.sign() == Sign::Minus {
            let lower_width = (self.lower.magnitude() - BigUint::one()).bits() + 1;
            let upper_width = {
                let upper = &self.upper - &BigInt::one();
                match upper.sign() {
                    Sign::Minus => (upper.magnitude() - BigUint::one()).bits() + 1,
                    Sign::NoSign => 0,
                    Sign::Plus => (upper.magnitude() + BigUint::one()).bits(),
                }
            };

            lower_width.max(upper_width) as u32
        } else {
            (&self.upper - &BigInt::one()).bits() as u32
        }
    }

    fn offset_bit_width(&self) -> u32 {
        (self.size() - BigInt::one()).bits() as u32
    }
}

#[cfg(test)]
pub mod test {
    use crate::{
        context::NativeContext, execution_result::ExecutionResult, executor::JitNativeExecutor,
        starknet_stub::StubSyscallHandler, utils::*, values::JitValue,
    };
    use cairo_lang_compiler::{
        compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
        project::setup_project, CompilerConfig,
    };
    use cairo_lang_filesystem::db::init_dev_corelib;
    use cairo_lang_sierra::{
        ids::FunctionId,
        program::Program,
        program::{FunctionSignature, GenFunction, StatementIdx},
    };
    use cairo_lang_starknet::starknet_plugin_suite;
    use pretty_assertions_sorted::assert_eq;
    use std::io::Write;
    use std::{env::var, fmt::Formatter, fs, path::Path};

    macro_rules! load_cairo {
        ( $( $program:tt )+ ) => {
            $crate::utils::test::load_cairo_str(stringify!($($program)+))
        };
    }
    macro_rules! load_starknet {
        ( $( $program:tt )+ ) => {
            $crate::utils::test::load_starknet_str(stringify!($($program)+))
        };
    }
    pub(crate) use load_cairo;
    pub(crate) use load_starknet;

    // Helper macros for faster testing.
    macro_rules! jit_struct {
        ($($y:expr),* $(,)? ) => {
            crate::values::JitValue::Struct {
                fields: vec![$($y), *],
                debug_name: None
            }
        };
    }
    macro_rules! jit_enum {
        ( $tag:expr, $value:expr ) => {
            crate::values::JitValue::Enum {
                tag: $tag,
                value: Box::new($value),
                debug_name: None,
            }
        };
    }
    macro_rules! jit_dict {
        ( $($key:expr $(=>)+ $value:expr),* $(,)? ) => {
            crate::values::JitValue::Felt252Dict {
                value: {
                    let mut map = std::collections::HashMap::new();
                    $(map.insert($key.into(), $value.into());)*
                    map
                },
                debug_name: None,
            }
        };
    }
    macro_rules! jit_panic {
        ( $($value:expr)? ) => {
            crate::utils::test::jit_enum!(1, crate::utils::test::jit_struct!(
                crate::utils::test::jit_struct!(),
                [$($value), *].into()
            ))
        };
    }
    pub(crate) use jit_dict;
    pub(crate) use jit_enum;
    pub(crate) use jit_panic;
    pub(crate) use jit_struct;

    pub(crate) fn load_cairo_str(program_str: &str) -> (String, Program) {
        compile_program(program_str, RootDatabase::default())
    }

    pub(crate) fn load_starknet_str(program_str: &str) -> (String, Program) {
        compile_program(
            program_str,
            RootDatabase::builder()
                .with_plugin_suite(starknet_plugin_suite())
                .build()
                .unwrap(),
        )
    }

    pub(crate) fn compile_program(program_str: &str, mut db: RootDatabase) -> (String, Program) {
        let mut program_file = tempfile::Builder::new()
            .prefix("test_")
            .suffix(".cairo")
            .tempfile()
            .unwrap();
        fs::write(&mut program_file, program_str).unwrap();

        init_dev_corelib(
            &mut db,
            Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
        );
        let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
        let sierra_program_with_dbg = compile_prepared_db(
            &mut db,
            main_crate_ids,
            CompilerConfig {
                diagnostics_reporter: DiagnosticsReporter::stderr(),
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap();

        let module_name = program_file.path().with_extension("");
        let module_name = module_name.file_name().unwrap().to_str().unwrap();
        (module_name.to_string(), sierra_program_with_dbg.program)
    }

    pub fn run_program(
        program: &(String, Program),
        entry_point: &str,
        args: &[JitValue],
    ) -> ExecutionResult {
        let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
        let program = &program.1;

        let entry_point_id = &program
            .funcs
            .iter()
            .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
            .expect("Test program entry point not found.")
            .id;

        let context = NativeContext::new();

        let module = context
            .compile(program, None)
            .expect("Could not compile test program to MLIR.");

        // FIXME: There are some bugs with non-zero LLVM optimization levels.
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::None);
        executor
            .invoke_dynamic_with_syscall_handler(
                entry_point_id,
                args,
                Some(u128::MAX),
                &mut StubSyscallHandler::default(),
            )
            .unwrap()
    }

    #[track_caller]
    pub fn run_program_assert_output(
        program: &(String, Program),
        entry_point: &str,
        args: &[JitValue],
        output: JitValue,
    ) {
        let result = run_program(program, entry_point, args);
        assert_eq!(result.return_value, output);
    }

    // ==============================
    // == TESTS: get_integer_layout
    // ==============================
    /// Ensures that the host's `u8` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u8() {
        assert_eq!(get_integer_layout(8).align(), 1);
    }

    /// Ensures that the host's `u16` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u16() {
        assert_eq!(get_integer_layout(16).align(), 2);
    }

    /// Ensures that the host's `u32` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u32() {
        assert_eq!(get_integer_layout(32).align(), 4);
    }

    /// Ensures that the host's `u64` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u64() {
        assert_eq!(get_integer_layout(64).align(), 8);
    }

    /// Ensures that the host's `u128` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u128() {
        assert_eq!(get_integer_layout(128).align(), 16);
    }

    /// Ensures that the host's `u256` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u256() {
        assert_eq!(get_integer_layout(256).align(), 16);
    }

    /// Ensures that the host's `u512` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u512() {
        assert_eq!(get_integer_layout(512).align(), 16);
    }

    /// Ensures that the host's `Felt` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_felt() {
        assert_eq!(get_integer_layout(252).align(), 16);
    }

    // ==============================
    // == TESTS: find_entry_point
    // ==============================
    #[test]
    fn test_find_entry_point_with_empty_program() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![],
        };
        let entry_point = find_entry_point(&program, "entry_point");
        assert!(entry_point.is_none());
    }

    #[test]
    fn test_entry_point_not_found() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![GenFunction {
                id: FunctionId {
                    id: 0,
                    debug_name: Some("not_entry_point".into()),
                },
                signature: FunctionSignature {
                    ret_types: vec![],
                    param_types: vec![],
                },
                params: vec![],
                entry_point: StatementIdx(0),
            }],
        };
        let entry_point = find_entry_point(&program, "entry_point");
        assert!(entry_point.is_none());
    }

    #[test]
    fn test_entry_point_found() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![GenFunction {
                id: FunctionId {
                    id: 0,
                    debug_name: Some("entry_point".into()),
                },
                signature: FunctionSignature {
                    ret_types: vec![],
                    param_types: vec![],
                },
                params: vec![],
                entry_point: StatementIdx(0),
            }],
        };
        let entry_point = find_entry_point(&program, "entry_point");
        assert!(entry_point.is_some());
        assert_eq!(entry_point.unwrap().id.id, 0);
    }

    // ====================================
    // == TESTS: find_entry_point_by_idx
    // ====================================
    #[test]
    fn test_find_entry_point_by_idx_with_empty_program() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![],
        };
        let entry_point = find_entry_point_by_idx(&program, 0);
        assert!(entry_point.is_none());
    }

    #[test]
    fn test_entry_point_not_found_by_id() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![GenFunction {
                id: FunctionId {
                    id: 0,
                    debug_name: Some("some_name".into()),
                },
                signature: FunctionSignature {
                    ret_types: vec![],
                    param_types: vec![],
                },
                params: vec![],
                entry_point: StatementIdx(0),
            }],
        };
        let entry_point = find_entry_point_by_idx(&program, 1);
        assert!(entry_point.is_none());
    }

    #[test]
    fn test_entry_point_found_by_id() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![GenFunction {
                id: FunctionId {
                    id: 15,
                    debug_name: Some("some_name".into()),
                },
                signature: FunctionSignature {
                    ret_types: vec![],
                    param_types: vec![],
                },
                params: vec![],
                entry_point: StatementIdx(0),
            }],
        };
        let entry_point = find_entry_point_by_idx(&program, 15);
        assert!(entry_point.is_some());
        assert_eq!(entry_point.unwrap().id.id, 15);
    }

    // ==============================
    // == TESTS: felt252_str
    // ==============================
    #[test]
    #[should_panic(expected = "value must be a digit number")]
    fn test_felt252_str_invalid_input() {
        let value = "not_a_number";
        felt252_str(value);
    }

    #[test]
    fn test_felt252_str_positive_number() {
        let value = "123";
        let result = felt252_str(value);
        assert_eq!(result, [123, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_felt252_str_negative_number() {
        let value = "-123";
        let result = felt252_str(value);
        assert_eq!(
            result,
            [
                4294967174, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 16,
                134217728
            ]
        );
    }

    #[test]
    fn test_felt252_str_zero() {
        let value = "0";
        let result = felt252_str(value);
        assert_eq!(result, [0, 0, 0, 0, 0, 0, 0, 0]);
    }

    // ==============================
    // == TESTS: felt252_short_str
    // ==============================
    #[test]
    fn test_felt252_short_str_short_numeric_string() {
        let value = "12345";
        let result = felt252_short_str(value);
        assert_eq!(result, [842216501, 49, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_felt252_short_str_short_string_with_non_numeric_characters() {
        let value = "hello";
        let result = felt252_short_str(value);
        assert_eq!(result, [1701604463, 104, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_felt252_short_str_long_numeric_string() {
        let value = "1234567890123456789012345678901234567890";
        let result = felt252_short_str(value);
        assert_eq!(
            result,
            [
                926431536, 859059510, 959459634, 892745528, 825373492, 926431536, 859059510,
                959459634
            ]
        );
    }

    #[test]
    fn test_felt252_short_str_empty_string() {
        let value = "";
        let result = felt252_short_str(value);
        assert_eq!(result, [0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_felt252_short_str_string_with_non_ascii_characters() {
        let value = "h€llø";
        let result = felt252_short_str(value);
        assert_eq!(result, [6843500, 0, 0, 0, 0, 0, 0, 0]);
    }

    // ==============================
    // == TESTS: debug_with
    // ==============================
    #[test]
    fn test_debug_with_empty_closure() {
        let closure = |_f: &mut Formatter| -> fmt::Result { Ok(()) };
        let debug_wrapper = debug_with(closure);
        assert_eq!(format!("{:?}", debug_wrapper), "");
    }

    #[test]
    #[should_panic]
    fn test_debug_with_error_closure() {
        let closure = |_f: &mut Formatter| -> Result<(), fmt::Error> { Err(fmt::Error) };
        let debug_wrapper = debug_with(closure);
        let _ = format!("{:?}", debug_wrapper);
    }

    #[test]
    fn test_debug_with_simple_closure() {
        let closure = |f: &mut fmt::Formatter| write!(f, "Hello, world!");
        let debug_wrapper = debug_with(closure);
        assert_eq!(format!("{:?}", debug_wrapper), "Hello, world!");
    }

    #[test]
    fn test_debug_with_complex_closure() {
        let closure = |f: &mut fmt::Formatter| write!(f, "Name: William, Age: {}", 28);
        let debug_wrapper = debug_with(closure);
        assert_eq!(format!("{:?}", debug_wrapper), "Name: William, Age: 28");
    }

    #[test]
    fn test_generate_function_name_debug_name() {
        let function_id = FunctionId {
            id: 123,
            debug_name: Some("function_name".into()),
        };

        assert_eq!(generate_function_name(&function_id), "function_name(f123)");
    }

    #[test]
    fn test_generate_function_name_without_debug_name() {
        let function_id = FunctionId {
            id: 123,
            debug_name: None,
        };

        assert_eq!(generate_function_name(&function_id), "f123");
    }

    #[test]
    fn test_cairo_to_sierra_path() {
        // Define the path to the cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the cairo program to sierra.
        let sierra_program = cairo_to_sierra(program_path);

        // Define the entry point function for comparison.
        let entry_point = "hello::hello::greet";
        // Find the function ID of the entry point function in the sierra program.
        let entry_point_id = find_function_id(&sierra_program, entry_point);

        // Assert that the debug name of the entry point function matches the expected value.
        assert_eq!(
            entry_point_id.debug_name,
            Some("hello::hello::greet".into())
        );
    }

    #[test]
    fn test_cairo_to_sierra_source() {
        // Define the content of the cairo program as a string.
        let content = "type u8 = u8;";

        // Create a named temporary file and write the content to it.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        // Get the path of the temporary file.
        let file_path = file.path().to_path_buf();

        // Compile the cairo program to sierra using the path of the temporary file.
        let sierra_program = cairo_to_sierra(&file_path);

        // Assert that the sierra program has no library function declarations, statements, or functions.
        assert!(sierra_program.libfunc_declarations.is_empty());
        assert!(sierra_program.statements.is_empty());
        assert!(sierra_program.funcs.is_empty());

        // Assert that the debug name of the first type declaration matches the expected value.
        assert_eq!(sierra_program.type_declarations.len(), 1);
        assert_eq!(
            sierra_program.type_declarations[0].id.debug_name,
            Some("u8".into())
        );
    }

    #[test]
    fn test_cairo_to_sierra_with_debug_info() {
        // Define the path to the cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Create a new context.
        let context = Context::new();
        // Compile the cairo program to sierra, including debug information.
        let sierra_program = cairo_to_sierra_with_debug_info(&context, program_path).unwrap();

        // Define the name of the entry point function for comparison.
        let entry_point = "hello::hello::greet";
        // Find the function ID of the entry point function in the sierra program.
        let entry_point_id = find_function_id(&sierra_program.0, entry_point);

        // Assert that the debug name of the entry point function matches the expected value.
        assert_eq!(
            entry_point_id.debug_name,
            Some("hello::hello::greet".into())
        );

        // Check if the sierra program contains a function with the specified debug name for the entry point function.
        assert!(sierra_program
            .1
            .funcs
            .keys()
            .any(|func| func.debug_name == Some("hello::hello::greet".into())));
    }
}
