//! # Various utilities

use cairo_felt::Felt252;
use cairo_lang_compiler::CompilerConfig;
use cairo_lang_sierra::{
    ids::FunctionId,
    program::{GenFunction, Program, StatementIdx},
};
use melior::{ir::Module, ExecutionEngine};
use num_bigint::{BigInt, BigUint, Sign};
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

/// Generate a function name.
///
/// If the program includes function identifiers, return those. Otherwise return `f` followed by the
/// identifier number.
pub fn generate_function_name(function_id: &FunctionId) -> Cow<str> {
    function_id
        .debug_name
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(format!("f{}", function_id.id)))
}

/// Return the layout for an integer of arbitrary width.
///
/// This assumes the platform's maximum (effective) alignment is 8 bytes, and that every integer
/// with a size in bytes of a power of two has the same alignment as its size.
pub fn get_integer_layout(width: u32) -> Layout {
    // TODO: Fix integer layouts properly.
    if width == 252 {
        #[cfg(target_arch = "x86_64")]
        return Layout::from_size_align(32, 8).unwrap();
        #[cfg(not(target_arch = "x86_64"))]
        return Layout::from_size_align(32, 16).unwrap();
    }

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
        Layout::new::<u128>()
    } else {
        Layout::array::<u64>(next_multiple_of_u32(width, 64) as usize >> 6).unwrap()
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
    } else {
        let source = std::fs::read_to_string(program).unwrap();
        Arc::new(
            cairo_lang_sierra::ProgramParser::new()
                .parse(&source)
                .unwrap(),
        )
    }
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

/// Given a string representing a function name, searches in the program for the id corresponding to said function, and returns a reference to it.
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

/// Converts a u32 slice into a Felt252
pub fn u32_vec_to_felt(u32_limbs: &[u32]) -> Felt252 {
    let mut ret = vec![];

    for limb in u32_limbs {
        let bytes = limb.to_le_bytes();
        ret.extend_from_slice(&bytes);
    }

    Felt252::from_bytes_le(&ret)
}

/// Creates the execution engine, with all symbols registered.
pub fn create_engine(module: &Module) -> ExecutionEngine {
    // Create the JIT engine.
    let engine = ExecutionEngine::new(module, 3, &[], false);

    #[cfg(feature = "with-runtime")]
    register_runtime_symbols(&engine);

    engine
}

#[cfg(feature = "with-runtime")]
pub fn register_runtime_symbols(engine: &ExecutionEngine) {
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
            cairo_native_runtime::cairo_native__alloc_dict as *const fn() -> *mut std::ffi::c_void
                as *mut (),
        );

        engine.register_symbol(
            "cairo_native__dict_get",
            cairo_native_runtime::cairo_native__dict_get
                as *const fn(*mut std::ffi::c_void, &[u8; 32]) -> *mut std::ffi::c_void
                as *mut (),
        );

        engine.register_symbol(
            "cairo_native__dict_insert",
            cairo_native_runtime::cairo_native__dict_insert
                as *const fn(
                    *mut std::ffi::c_void,
                    &[u8; 32],
                    NonNull<std::ffi::c_void>,
                ) -> *mut std::ffi::c_void as *mut (),
        );
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

/// The `mlir_asm!` macro is a shortcut to manually building operations.
///
/// It works by forwarding the custom DSL code to their respective functions within melior's
/// `OperationBuilder`.
///
/// The DSL's syntax is similar to that of MLIR, but has some differences, or rather restrictions,
/// due to the way declarative macros work:
///   - All macro invocations need the MLIR context, the target block and the operations' locations.
///   - The operations are defined using a syntax similar to that of MLIR's generic operations, with
///     some differences. The results are Rust variables (MLIR values) and the inputs (operands,
///     attributes...) are all Rust expressions that evaluate to their respective type.
///
/// Check out the [felt252 libfunc implementations](crate::libfuncs::felt252) for an example on their usage.
macro_rules! mlir_asm {
    (
        $context:expr, $block:expr, $location:expr =>
            $( ; $( $( $ret:ident ),+ = )? $op:literal
                ( $( $( $arg:expr ),+ $(,)? )? ) // Operands.
                $( [ $( $( ^ $successor:ident $( ( $( $( $successor_arg:expr ),+ $(,)? )? ) )? ),+ $(,)? )? ] )? // Successors.
                $( < { $( $( $prop_name:pat_param = $prop_value:expr ),+ $(,)? )? } > )? // Properties.
                $( ( $( $( $region:expr ),+ $(,)? )? ) )? // Regions.
                $( { $( $( $attr_name:literal = $attr_value:expr ),+ $(,)? )? } )? // Attributes.
                : $args_ty:tt -> $rets_ty:tt // Signature.
            )*
    ) => { $(
        #[allow(unused_mut)]
        $( let $crate::utils::codegen_ret_decl!($($ret),+) = )? {
            #[allow(unused_variables)]
            let context = $context;
            let mut builder = melior::ir::operation::OperationBuilder::new($op, $location);

            // Process operands.
            $( let builder = builder.add_operands(&[$( $arg, )+]); )?

            // TODO: Process successors.
            // TODO: Process properties.
            // TODO: Process regions.

            // Process attributes.
            $( $(
                let builder = $crate::utils::codegen_attributes!(context, builder => $($attr_name = $attr_value),+);
            )? )?

            // Process signature.
            // #[cfg(debug_assertions)]
            // $crate::utils::codegen_signature!( PARAMS $args_ty );
            let builder = $crate::utils::codegen_signature!( RETS builder => $rets_ty );

            #[allow(unused_variables)]
            let op = $block.append_operation(builder.build());
            $( $crate::utils::codegen_ret_extr!(op => $($ret),+) )?
        };
    )* };
}
pub(crate) use mlir_asm;

macro_rules! codegen_attributes {
    // Macro entry points.
    ( $context:ident, $builder:ident => $name:literal = $value:expr ) => {
        $builder.add_attributes(&[
            $crate::utils::codegen_attributes!(INTERNAL $context, $builder => $name = $value),
        ])
    };
    ( $context:ident, $builder:ident => $( $name:literal = $value:expr ),+ ) => {
        $builder.add_attributes(&[
            $( $crate::utils::codegen_attributes!(INTERNAL $context, $builder => $name = $value), )+
        ])
    };

    ( INTERNAL $context:ident, $builder:ident => $name:literal = $value:expr ) => {
        (
            melior::ir::Identifier::new($context, $name),
            $value,
        )
    };
}
pub(crate) use codegen_attributes;

macro_rules! codegen_signature {
    ( PARAMS ) => {
        // TODO: Check operand types.
    };

    ( RETS $builder:ident => () ) => { $builder };
    ( RETS $builder:ident => $ret_ty:expr ) => {
        $builder.add_results(&[$ret_ty])
    };
    ( RETS $builder:ident => $( $ret_ty:expr ),+ $(,)? ) => {
        $builder.add_results(&[$($ret_ty),+])
    };
}
pub(crate) use codegen_signature;

macro_rules! codegen_ret_decl {
    // Macro entry points.
    ( $ret:ident ) => { $ret };
    ( $( $ret:ident ),+ ) => {
        ( $( codegen_ret_decl!($ret) ),+ )
    };
}
pub(crate) use codegen_ret_decl;

macro_rules! codegen_ret_extr {
    // Macro entry points.
    ( $op:ident => $ret:ident ) => {{
        melior::ir::Value::from($op.result(0)?)
    }};
    ( $op:ident => $( $ret:ident ),+ ) => {{
        let mut idx = 0;
        ( $( codegen_ret_extr!(INTERNAL idx, $op => $ret) ),+ )
    }};

    // Internal entry points.
    ( INTERNAL $count:ident, $op:ident => $ret:ident ) => {
        {
            let idx = $count;
            $count += 1;
            melior::ir::Value::from($op.result(idx)?)
        }
    };
}
use crate::types::felt252::PRIME;
pub(crate) use codegen_ret_extr;

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::{
        metadata::{
            gas::{GasMetadata, MetadataComputationConfig},
            runtime_bindings::RuntimeBindingsMeta,
            MetadataStorage,
        },
        types::felt252::PRIME,
    };
    use cairo_lang_compiler::{
        compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
        project::setup_project, CompilerConfig,
    };
    use cairo_lang_filesystem::db::init_dev_corelib;
    use cairo_lang_sierra::{
        extensions::core::{CoreLibfunc, CoreType},
        program::Program,
        program_registry::ProgramRegistry,
    };
    use melior::{
        dialect::DialectRegistry,
        ir::{Location, Module},
        pass::{self, PassManager},
        utility::{register_all_dialects, register_all_passes},
        Context, ExecutionEngine,
    };
    use num_bigint::{BigInt, Sign};
    use std::{env::var, fs, ops::Neg, path::Path};

    macro_rules! load_cairo {
        ( $( $program:tt )+ ) => {
            $crate::utils::test::load_cairo_str(stringify!($($program)+))
        };
    }
    pub(crate) use load_cairo;

    pub fn load_cairo_str(program_str: &str) -> (String, Program) {
        let mut program_file = tempfile::Builder::new()
            .prefix("test_")
            .suffix(".cairo")
            .tempfile()
            .unwrap();
        fs::write(&mut program_file, program_str).unwrap();

        let mut db = RootDatabase::default();
        init_dev_corelib(
            &mut db,
            Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
        );
        let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
        let program = compile_prepared_db(
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
        (module_name.to_string(), (*program).clone())
    }

    pub fn run_program(
        program: &(String, Program),
        entry_point: &str,
        args: serde_json::Value,
    ) -> serde_json::Value {
        let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
        let program = &program.1;

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(program)
            .expect("Could not create the test program registry.");

        let entry_point_id = &program
            .funcs
            .iter()
            .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
            .expect("Test program entry point not found.")
            .id;

        let context = Context::new();
        context.append_dialect_registry(&{
            let registry = DialectRegistry::new();
            register_all_dialects(&registry);
            registry
        });
        context.load_all_available_dialects();
        register_all_passes();

        let mut module = Module::new(Location::unknown(&context));

        let mut metadata = MetadataStorage::new();

        // Make the runtime library available.
        metadata.insert(RuntimeBindingsMeta::default()).unwrap();

        // Gas
        let required_initial_gas = if program
            .type_declarations
            .iter()
            .any(|decl| decl.long_id.generic_id.0.as_str() == "GasBuiltin")
        {
            let gas_metadata = GasMetadata::new(program, MetadataComputationConfig::default());

            let required_initial_gas = { gas_metadata.get_initial_required_gas(entry_point_id) };
            metadata.insert(gas_metadata).unwrap();
            required_initial_gas
        } else {
            None
        };

        crate::compile::<CoreType, CoreLibfunc>(
            &context,
            &module,
            program,
            &registry,
            &mut metadata,
            None,
        )
        .expect("Could not compile test program to MLIR.");

        assert!(
            module.as_operation().verify(),
            "Test program generated invalid MLIR:\n{}",
            module.as_operation()
        );

        let pass_manager = PassManager::new(&context);
        pass_manager.enable_verifier(true);
        pass_manager.add_pass(pass::transform::create_canonicalizer());

        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());

        pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager.add_pass(pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

        pass_manager
            .run(&mut module)
            .expect("Could not apply passes to the compiled test program.");

        let engine = ExecutionEngine::new(&module, 0, &[], false);

        #[cfg(feature = "with-runtime")]
        register_runtime_symbols(&engine);

        crate::execute::<CoreType, CoreLibfunc, _, _>(
            &engine,
            &registry,
            &program
                .funcs
                .iter()
                .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
                .expect("Test program entry point not found.")
                .id,
            args,
            serde_json::value::Serializer,
            required_initial_gas,
        )
        .expect("Test program execution failed.")
    }

    // Parse numeric string into felt, wrapping negatives around the prime modulo.
    pub fn felt(value: &str) -> [u32; 8] {
        let value = value.parse::<BigInt>().unwrap();
        let value = match value.sign() {
            Sign::Minus => &*PRIME - value.neg().to_biguint().unwrap(),
            _ => value.to_biguint().unwrap(),
        };

        let mut u32_digits = value.to_u32_digits();
        u32_digits.resize(8, 0);
        u32_digits.try_into().unwrap()
    }

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
    #[ignore]
    fn test_alignment_compatibility_u128() {
        // FIXME: Uncomment once LLVM fixes its u128 alignment issues.
        assert_eq!(get_integer_layout(128).align(), 16);
    }

    /// Ensures that the host's `u256` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u256() {
        assert_eq!(get_integer_layout(256).align(), 8);
    }

    /// Ensures that the host's `u512` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u512() {
        assert_eq!(get_integer_layout(512).align(), 8);
    }

    /// Ensures that the host's `felt252` is compatible with its compiled counterpart.
    #[test]
    #[ignore]
    fn test_alignment_compatibility_felt252() {
        assert_eq!(get_integer_layout(252).align(), 8);
    }
}
