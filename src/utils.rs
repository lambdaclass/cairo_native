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
    extensions::core::{CoreLibfunc, CoreType},
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
        Arc::new(
            cairo_lang_sierra::ProgramParser::new()
                .parse(&source)
                .unwrap(),
        )
    }
}

pub fn cairo_to_sierra_with_debug_info<'ctx>(
    context: &'ctx Context,
    program: &Path,
) -> Result<(Program, DebugLocations<'ctx>), crate::error::Error> {
    let mut db = RootDatabase::builder().detect_corelib().build().unwrap();
    let main_crate_ids = setup_project(&mut db, program).unwrap();
    let program = compile_prepared_db(
        &mut db,
        main_crate_ids,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )
    .unwrap();

    let debug_locations = {
        let debug_info = DebugInfo::extract(&db, &program)
            .map_err(|_| {
                let mut buffer = String::new();
                assert!(DiagnosticsReporter::write_to_string(&mut buffer).check(&db));
                buffer
            })
            .unwrap();

        DebugLocations::extract(context, &db, &debug_info)
    };

    Ok((program, debug_locations))
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
            "cairo_native__dict_free",
            cairo_native_runtime::cairo_native__dict_free as *const fn(*mut std::ffi::c_void) -> ()
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

        engine.register_symbol(
            "cairo_native__dict_gas_refund",
            cairo_native_runtime::cairo_native__dict_gas_refund
                as *const fn(*const std::ffi::c_void, NonNull<std::ffi::c_void>) -> u64
                as *mut (),
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
            let op = $block.append_operation(builder.build()?);
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
pub(crate) use codegen_ret_extr;

#[cfg(test)]
pub mod test {
    use crate::{
        context::NativeContext,
        execution_result::ExecutionResult,
        executor::JitNativeExecutor,
        starknet::{
            BlockInfo, ExecutionInfo, ExecutionInfoV2, ResourceBounds, StarknetSyscallHandler,
            SyscallResult, TxInfo, TxV2Info, U256,
        },
        utils::*,
        values::JitValue,
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
    use starknet_types_core::felt::Felt;
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
        ( $($x:expr),* $(,)? ) => {
            crate::values::JitValue::Struct {
                fields: vec![$($x), *],
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
        (module_name.to_string(), program)
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
                TestSyscallHandler,
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

    #[derive(Debug, Clone)]
    pub struct TestSyscallHandler;

    impl StarknetSyscallHandler for TestSyscallHandler {
        fn get_block_hash(&mut self, block_number: u64, _gas: &mut u128) -> SyscallResult<Felt> {
            dbg!(block_number);
            Ok(Felt::from(block_number))
        }

        fn get_execution_info(
            &mut self,
            _gas: &mut u128,
        ) -> SyscallResult<crate::starknet::ExecutionInfo> {
            Ok(ExecutionInfo {
                block_info: BlockInfo {
                    block_number: 1234,
                    block_timestamp: 2345,
                    sequencer_address: 3456.into(),
                },
                tx_info: TxInfo {
                    version: 4567.into(),
                    account_contract_address: 5678.into(),
                    max_fee: 6789,
                    signature: vec![1248.into(), 2486.into()],
                    transaction_hash: 9876.into(),
                    chain_id: 8765.into(),
                    nonce: 7654.into(),
                },
                caller_address: 6543.into(),
                contract_address: 5432.into(),
                entry_point_selector: 4321.into(),
            })
        }

        fn get_execution_info_v2(
            &mut self,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<crate::starknet::ExecutionInfoV2> {
            Ok(ExecutionInfoV2 {
                block_info: BlockInfo {
                    block_number: 1234,
                    block_timestamp: 2345,
                    sequencer_address: 3456.into(),
                },
                tx_info: TxV2Info {
                    version: 1.into(),
                    account_contract_address: 1.into(),
                    max_fee: 0,
                    signature: vec![1.into()],
                    transaction_hash: 1.into(),
                    chain_id: 1.into(),
                    nonce: 1.into(),
                    tip: 1,
                    paymaster_data: vec![1.into()],
                    nonce_data_availability_mode: 0,
                    fee_data_availability_mode: 0,
                    account_deployment_data: vec![1.into()],
                    resource_bounds: vec![ResourceBounds {
                        resource: 2.into(),
                        max_amount: 10,
                        max_price_per_unit: 20,
                    }],
                },
                caller_address: 6543.into(),
                contract_address: 5432.into(),
                entry_point_selector: 4321.into(),
            })
        }

        fn deploy(
            &mut self,
            class_hash: Felt,
            contract_address_salt: Felt,
            calldata: &[Felt],
            _deploy_from_zero: bool,
            _gas: &mut u128,
        ) -> SyscallResult<(Felt, Vec<Felt>)> {
            Ok((
                class_hash + contract_address_salt,
                calldata.iter().map(|x| x + Felt::ONE).collect(),
            ))
        }

        fn replace_class(&mut self, _class_hash: Felt, _gas: &mut u128) -> SyscallResult<()> {
            Ok(())
        }

        fn library_call(
            &mut self,
            _class_hash: Felt,
            _function_selector: Felt,
            calldata: &[Felt],
            _gas: &mut u128,
        ) -> SyscallResult<Vec<Felt>> {
            Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
        }

        fn call_contract(
            &mut self,
            _address: Felt,
            _entry_point_selector: Felt,
            calldata: &[Felt],
            _gas: &mut u128,
        ) -> SyscallResult<Vec<Felt>> {
            Ok(calldata.iter().map(|x| x * Felt::from(3)).collect())
        }

        fn storage_read(
            &mut self,
            _address_domain: u32,
            address: Felt,
            _gas: &mut u128,
        ) -> SyscallResult<Felt> {
            Ok(address * Felt::from(3))
        }

        fn storage_write(
            &mut self,
            _address_domain: u32,
            _address: Felt,
            _value: Felt,
            _gas: &mut u128,
        ) -> SyscallResult<()> {
            Ok(())
        }

        fn emit_event(
            &mut self,
            _keys: &[Felt],
            _data: &[Felt],
            _gas: &mut u128,
        ) -> SyscallResult<()> {
            Ok(())
        }

        fn send_message_to_l1(
            &mut self,
            _to_address: Felt,
            _payload: &[Felt],
            _gas: &mut u128,
        ) -> SyscallResult<()> {
            Ok(())
        }

        fn keccak(
            &mut self,
            _input: &[u64],
            gas: &mut u128,
        ) -> SyscallResult<crate::starknet::U256> {
            *gas -= 1000;
            Ok(U256 {
                hi: 0,
                lo: 1234567890,
            })
        }

        // Implementing the secp256 syscalls for testing doesn't make sense. They're already
        // properly tested in integration tests.

        fn secp256k1_new(
            &mut self,
            _x: U256,
            _y: U256,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<Option<crate::starknet::Secp256k1Point>> {
            unimplemented!()
        }

        fn secp256k1_add(
            &mut self,
            _p0: crate::starknet::Secp256k1Point,
            _p1: crate::starknet::Secp256k1Point,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<crate::starknet::Secp256k1Point> {
            unimplemented!()
        }

        fn secp256k1_mul(
            &mut self,
            _p: crate::starknet::Secp256k1Point,
            _m: U256,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<crate::starknet::Secp256k1Point> {
            unimplemented!()
        }

        fn secp256k1_get_point_from_x(
            &mut self,
            _x: U256,
            _y_parity: bool,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<Option<crate::starknet::Secp256k1Point>> {
            unimplemented!()
        }

        fn secp256k1_get_xy(
            &mut self,
            _p: crate::starknet::Secp256k1Point,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<(U256, U256)> {
            unimplemented!()
        }

        fn secp256r1_new(
            &mut self,
            _x: U256,
            _y: U256,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<Option<crate::starknet::Secp256r1Point>> {
            unimplemented!()
        }

        fn secp256r1_add(
            &mut self,
            _p0: crate::starknet::Secp256r1Point,
            _p1: crate::starknet::Secp256r1Point,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<crate::starknet::Secp256r1Point> {
            unimplemented!()
        }

        fn secp256r1_mul(
            &mut self,
            _p: crate::starknet::Secp256r1Point,
            _m: U256,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<crate::starknet::Secp256r1Point> {
            unimplemented!()
        }

        fn secp256r1_get_point_from_x(
            &mut self,
            _x: U256,
            _y_parity: bool,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<Option<crate::starknet::Secp256r1Point>> {
            unimplemented!()
        }

        fn secp256r1_get_xy(
            &mut self,
            _p: crate::starknet::Secp256r1Point,
            _remaining_gas: &mut u128,
        ) -> SyscallResult<(U256, U256)> {
            unimplemented!()
        }
    }
}
