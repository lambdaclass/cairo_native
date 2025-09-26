//! # Various utilities

pub(crate) use self::{program_registry_ext::ProgramRegistryExt, range_ext::RangeExt};
use crate::{
    error::Result as NativeResult, metadata::MetadataStorage, native_panic, types::TypeBuilder,
    OptLevel,
};
use cairo_lang_runner::token_gas_cost;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        gas::CostTokenType,
    },
    ids::{ConcreteTypeId, FunctionId},
    program::{GenFunction, Program, StatementIdx},
    program_registry::ProgramRegistry,
};
use melior::{
    ir::Module,
    pass::{self, PassManager},
    Context, Error, ExecutionEngine,
};
use num_bigint::{BigInt, BigUint, Sign};
use serde::{Deserialize, Serialize};
use starknet_types_core::felt::Felt;
use std::sync::LazyLock;
use std::{
    alloc::Layout,
    borrow::Cow,
    fmt::{self, Display},
};
use thiserror::Error;

pub mod mem_tracing;
mod program_registry_ext;
mod range_ext;
#[cfg(feature = "with-segfault-catcher")]
pub mod safe_runner;
pub mod sierra_gen;
#[cfg(any(feature = "testing", test))]
pub mod testing;
pub mod trace_dump;
pub mod walk_ir;

#[cfg(target_os = "macos")]
pub const SHARED_LIBRARY_EXT: &str = "dylib";
#[cfg(target_os = "linux")]
pub const SHARED_LIBRARY_EXT: &str = "so";

/// The `felt252` prime modulo.
pub static PRIME: LazyLock<BigUint> = LazyLock::new(|| {
    "3618502788666131213697322783095070105623107215331596699973092056135872020481"
        .parse()
        .expect("hardcoded prime constant should be valid")
});
pub static HALF_PRIME: LazyLock<BigUint> = LazyLock::new(|| {
    "1809251394333065606848661391547535052811553607665798349986546028067936010240"
        .parse()
        .expect("hardcoded half prime constant should be valid")
});

/// Represents the gas cost of each cost token type
///
/// See `crate::metadata::gas` for more documentation.
///
/// Order matters, for the libfunc impl
/// https://github.com/starkware-libs/sequencer/blob/1b7252f8a30244d39614d7666aa113b81291808e/crates/blockifier/src/execution/entry_point_execution.rs#L208
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[repr(C)]
pub struct BuiltinCosts {
    pub r#const: u64,
    pub pedersen: u64,
    pub bitwise: u64,
    pub ecop: u64,
    pub poseidon: u64,
    pub add_mod: u64,
    pub mul_mod: u64,
}

impl BuiltinCosts {
    pub fn index_for_token_type(token_type: &CostTokenType) -> NativeResult<usize> {
        let index = match token_type {
            CostTokenType::Const => 0,
            CostTokenType::Pedersen => 1,
            CostTokenType::Bitwise => 2,
            CostTokenType::EcOp => 3,
            CostTokenType::Poseidon => 4,
            CostTokenType::AddMod => 5,
            CostTokenType::MulMod => 6,
            _ => native_panic!("matched an unexpected CostTokenType which is not being used"),
        };

        Ok(index)
    }
}

impl Default for BuiltinCosts {
    fn default() -> Self {
        Self {
            r#const: token_gas_cost(CostTokenType::Const) as u64,
            pedersen: token_gas_cost(CostTokenType::Pedersen) as u64,
            bitwise: token_gas_cost(CostTokenType::Bitwise) as u64,
            ecop: token_gas_cost(CostTokenType::EcOp) as u64,
            poseidon: token_gas_cost(CostTokenType::Poseidon) as u64,
            add_mod: token_gas_cost(CostTokenType::AddMod) as u64,
            mul_mod: token_gas_cost(CostTokenType::MulMod) as u64,
        }
    }
}

impl crate::arch::AbiArgument for BuiltinCosts {
    fn to_bytes(
        &self,
        buffer: &mut Vec<u8>,
        find_dict_drop_override: impl Copy
            + Fn(
                &cairo_lang_sierra::ids::ConcreteTypeId,
            ) -> Option<extern "C" fn(*mut std::ffi::c_void)>,
    ) -> crate::error::Result<()> {
        self.r#const.to_bytes(buffer, find_dict_drop_override)?;
        self.pedersen.to_bytes(buffer, find_dict_drop_override)?;
        self.bitwise.to_bytes(buffer, find_dict_drop_override)?;
        self.ecop.to_bytes(buffer, find_dict_drop_override)?;
        self.poseidon.to_bytes(buffer, find_dict_drop_override)?;
        self.add_mod.to_bytes(buffer, find_dict_drop_override)?;
        self.mul_mod.to_bytes(buffer, find_dict_drop_override)?;

        Ok(())
    }
}

#[cfg(feature = "with-mem-tracing")]
#[allow(unused_imports)]
pub(crate) use self::mem_tracing::{
    _wrapped_free as libc_free, _wrapped_malloc as libc_malloc, _wrapped_realloc as libc_realloc,
};
#[cfg(not(feature = "with-mem-tracing"))]
#[allow(unused_imports)]
pub(crate) use libc::{free as libc_free, malloc as libc_malloc, realloc as libc_realloc};

/// Generate a function name.
///
/// If the program includes function identifiers, return those. Otherwise return `f` followed by the
/// identifier number.
pub fn generate_function_name(
    function_id: &'_ FunctionId,
    is_for_contract_executor: bool,
) -> Cow<'_, str> {
    // Generic functions can omit their type in the debug_name, leading to multiple functions
    // having the same name, we solve this by adding the id number even if the function has a debug_name

    if is_for_contract_executor {
        Cow::Owned(format!("f{}", function_id.id))
    } else if let Some(name) = function_id.debug_name.as_deref() {
        Cow::Owned(format!("{}(f{})", name, function_id.id))
    } else {
        Cow::Owned(format!("f{}", function_id.id))
    }
}

/// Decode an UTF-8 error message replacing invalid bytes with their hexadecimal representation, as
/// done by Python's `x.decode('utf-8', errors='backslashreplace')`.
pub fn decode_error_message(data: &[u8]) -> String {
    let mut pos = 0;
    utf8_iter::ErrorReportingUtf8Chars::new(data).fold(String::new(), |mut acc, ch| {
        match ch {
            Ok(ch) => {
                acc.push(ch);
                pos += ch.len_utf8();
            }
            Err(_) => {
                acc.push_str(&format!("\\x{:02x}", data[pos]));
                pos += 1;
            }
        };

        acc
    })
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
        Layout::new::<u128>()
    } else {
        // According to the docs this should never return an error.
        Layout::from_size_align((width as usize).next_multiple_of(8) >> 3, 16)
            .expect("layout size rounded up to the next multiple of 16 should never be greater than ISIZE::MAX")
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

/// Given a string representing a function name, searches in the program for the id corresponding
/// to said function, and returns a reference to it.
#[track_caller]
pub fn find_function_id<'a>(program: &'a Program, function_name: &str) -> Option<&'a FunctionId> {
    program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(function_name))
        .map(|func| &func.id)
}

/// Parse a numeric string into felt, wrapping negatives around the prime modulo.
pub fn felt252_str(value: &str) -> Felt {
    let value = value
        .parse::<BigInt>()
        .expect("value must be a digit number");

    let value = match value.sign() {
        Sign::Minus => &*PRIME - value.magnitude(),
        _ => value.magnitude().clone(),
    };

    value.into()
}

/// Parse any type that can be a bigint to a felt that can be used in the cairo-native input.
pub fn felt252_bigint(value: impl Into<BigInt>) -> Felt {
    let value: BigInt = value.into();
    let value = match value.sign() {
        Sign::Minus => Cow::Owned(&*PRIME - value.magnitude()),
        _ => Cow::Borrowed(value.magnitude()),
    };

    value.as_ref().into()
}

/// Parse a short string into a felt that can be used in the cairo-native input.
pub fn felt252_short_str(value: &str) -> Felt {
    let values: Vec<_> = value
        .chars()
        .filter_map(|c| c.is_ascii().then_some(c as u8))
        .collect();

    assert!(values.len() < 32, "A felt can't longer than 32 bytes");
    Felt::from_bytes_be_slice(&values)
}

/// Creates the execution engine, with all symbols registered.
pub fn create_engine(
    module: &Module,
    _metadata: &MetadataStorage,
    opt_level: OptLevel,
) -> ExecutionEngine {
    // Create the JIT engine.
    let engine = ExecutionEngine::new(module, opt_level.into(), &[], false);

    #[cfg(feature = "with-mem-tracing")]
    self::mem_tracing::register_bindings(&engine);

    engine
}

pub fn run_pass_manager(context: &Context, module: &mut Module) -> Result<(), Error> {
    let pass_manager = PassManager::new(context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow()); // needed because to_llvm doesn't include it.
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.run(module)
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

/// Returns the total layout size for the given types.
pub fn get_types_total_size(
    types_ids: &[ConcreteTypeId],
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
) -> crate::error::Result<usize> {
    let mut total_size = 0;
    for type_id in types_ids {
        let type_concrete = registry.get_type(type_id)?;
        let layout = type_concrete.layout(registry)?;
        total_size += layout.size();
    }
    Ok(total_size)
}
