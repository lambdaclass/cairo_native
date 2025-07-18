#![allow(non_snake_case)]

use crate::utils::BuiltinCosts;
use cairo_lang_sierra_gas::core_libfunc_cost::{
    DICT_SQUASH_REPEATED_ACCESS_COST, DICT_SQUASH_UNIQUE_KEY_COST,
};
use itertools::Itertools;
use lazy_static::lazy_static;
use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use rand::Rng;
use starknet_curve::curve_params::BETA;
use starknet_types_core::{
    curve::{AffinePoint, ProjectivePoint},
    felt::Felt,
    hash::StarkHash,
};
use std::{
    alloc::{dealloc, realloc, Layout},
    cell::Cell,
    collections::{hash_map::Entry, HashMap},
    ffi::{c_int, c_void},
    fs::File,
    io::Write,
    mem::{forget, ManuallyDrop},
    ops::Shl,
    os::fd::FromRawFd,
    ptr,
    rc::Rc,
};
use std::{ops::Mul, vec::IntoIter};

lazy_static! {
    pub static ref HALF_PRIME: Felt = Felt::from_dec_str(
        "1809251394333065606848661391547535052811553607665798349986546028067936010240"
    )
    .unwrap();
    pub static ref DICT_GAS_REFUND_PER_ACCESS: u64 =
        (DICT_SQUASH_UNIQUE_KEY_COST.cost() - DICT_SQUASH_REPEATED_ACCESS_COST.cost()) as u64;
}

/// Based on `cairo-lang-runner`'s implementation.
///
/// Source: <https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-runner/src/casm_run/mod.rs#L1946-L1948>
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__debug__print(
    target_fd: i32,
    data: *const [u8; 32],
    len: u32,
) -> i32 {
    // Avoid closing `stdout` on all branches.
    let mut target = ManuallyDrop::new(File::from_raw_fd(target_fd));

    let mut items = Vec::with_capacity(len as usize);

    for i in 0..len as usize {
        let mut data = *data.add(i);
        data[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

        let value = Felt::from_bytes_le(&data);
        items.push(value);
    }

    let value = format_for_debug(items.into_iter());

    if write!(target, "{}", value).is_err() {
        return 1;
    };

    0
}

/// Compute `pedersen(lhs, rhs)` and store it into `dst`.
///
/// All its operands need the values in big endian.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__pedersen(
    dst: &mut [u8; 32],
    lhs: &[u8; 32],
    rhs: &[u8; 32],
) {
    // Extract arrays from the pointers.
    let mut lhs = *lhs;
    let mut rhs = *rhs;

    lhs[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    rhs[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    // Convert to FieldElement.
    let lhs = Felt::from_bytes_le(&lhs);
    let rhs = Felt::from_bytes_le(&rhs);

    // Compute pedersen hash and copy the result into `dst`.
    let res = starknet_types_core::hash::Pedersen::hash(&lhs, &rhs);
    *dst = res.to_bytes_le();
}

/// Compute `hades_permutation(op0, op1, op2)` and replace the operands with the results.
///
/// All operands need the values in big endian.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__hades_permutation(
    op0: &mut [u8; 32],
    op1: &mut [u8; 32],
    op2: &mut [u8; 32],
) {
    op0[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    op1[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    op2[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    // Convert to FieldElement.
    let mut state = [
        Felt::from_bytes_le(op0),
        Felt::from_bytes_le(op1),
        Felt::from_bytes_le(op2),
    ];

    // Compute Poseidon permutation.
    starknet_types_core::hash::Poseidon::hades_permutation(&mut state);

    // Write back the results.
    *op0 = state[0].to_bytes_le();
    *op1 = state[1].to_bytes_le();
    *op2 = state[2].to_bytes_le();
}

/// Felt252 type used in cairo native runtime
#[derive(Debug)]
pub struct FeltDict {
    pub mappings: HashMap<[u8; 32], usize>,

    pub layout: Layout,
    pub elements: *mut (),

    pub drop_fn: Option<extern "C" fn(*mut c_void)>,

    pub count: u64,
}

impl Drop for FeltDict {
    fn drop(&mut self) {
        // Free the entries manually.
        if let Some(drop_fn) = self.drop_fn {
            for (_, &index) in self.mappings.iter() {
                let value_ptr = unsafe {
                    self.elements
                        .byte_add(self.layout.pad_to_align().size() * index)
                };

                drop_fn(value_ptr.cast());
            }
        }

        // Free the value data.
        if !self.elements.is_null() {
            unsafe {
                dealloc(
                    self.elements.cast(),
                    Layout::from_size_align_unchecked(
                        self.layout.pad_to_align().size() * self.mappings.capacity(),
                        self.layout.align(),
                    ),
                )
            };
        }
    }
}

/// Allocate a new dictionary.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__dict_new(
    size: u64,
    align: u64,
    drop_fn: Option<extern "C" fn(*mut c_void)>,
) -> *const FeltDict {
    Rc::into_raw(Rc::new(FeltDict {
        mappings: HashMap::default(),

        layout: Layout::from_size_align_unchecked(size as usize, align as usize),
        elements: ptr::null_mut(),

        drop_fn,

        count: 0,
    }))
}

/// Free a dictionary using an optional callback to drop each element.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
// Note: Using `Option<extern "C" fn(*mut c_void)>` is ffi-safe thanks to Option's null
//   pointer optimization. Check out
//   https://doc.rust-lang.org/nomicon/ffi.html#the-nullable-pointer-optimization for more info.
pub unsafe extern "C" fn cairo_native__dict_drop(ptr: *const FeltDict) {
    drop(Rc::from_raw(ptr));
}

/// Duplicate a dictionary using a provided callback to clone each element.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__dict_dup(dict_ptr: *const FeltDict) -> *const FeltDict {
    let old_dict = Rc::from_raw(dict_ptr);
    let new_dict = Rc::clone(&old_dict);

    forget(old_dict);
    Rc::into_raw(new_dict)
}

/// Return a pointer to the entry's value pointer for a given key, inserting a null pointer if not
/// present. Increment the access count.
///
/// The null pointer will be either updated by `felt252_dict_entry_finalize` or removed (along with
/// everything else in the dict) by the entry's drop implementation.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__dict_get(
    dict_ptr: *const FeltDict,
    key: &[u8; 32],
    value_ptr: *mut *mut c_void,
) -> c_int {
    let dict_rc = Rc::from_raw(dict_ptr);

    // there may me multiple reference to the same dictionary (snapshots), but
    // as snapshots cannot access the inner dictionary, then it is safe to modify it
    // without cloning it.
    let dict = Rc::as_ptr(&dict_rc)
        .cast_mut()
        .as_mut()
        .expect("rc inner pointer should never be null");

    let num_mappings = dict.mappings.len();
    let has_capacity = num_mappings != dict.mappings.capacity();

    let (is_present, index) = match dict.mappings.entry(*key) {
        Entry::Occupied(entry) => (true, *entry.get()),
        Entry::Vacant(entry) => {
            entry.insert(num_mappings);
            (false, num_mappings)
        }
    };

    // Maybe realloc (conditions: !has_capacity && !is_present).
    if !has_capacity && !is_present {
        dict.elements = realloc(
            dict.elements.cast(),
            Layout::from_size_align_unchecked(
                dict.layout.pad_to_align().size() * dict.mappings.len(),
                dict.layout.align(),
            ),
            dict.layout.pad_to_align().size() * dict.mappings.capacity(),
        )
        .cast();
    }

    *value_ptr = dict
        .elements
        .byte_add(dict.layout.pad_to_align().size() * index)
        .cast();

    dict.count += 1;
    forget(dict_rc);

    is_present as c_int
}

/// Simulates the felt252_dict_squash libfunc.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__dict_squash(
    dict_ptr: *const FeltDict,
    range_check_ptr: &mut u64,
    gas_ptr: &mut u64,
) {
    let dict = Rc::from_raw(dict_ptr);

    *gas_ptr +=
        (dict.count.saturating_sub(dict.mappings.len() as u64)) * *DICT_GAS_REFUND_PER_ACCESS;

    // Squashing a dictionary always uses the range check builtin at least twice.
    // https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/felt252_dict.rs?plain=1#L131-L136
    *range_check_ptr += 2;

    let no_big_keys = dict
        .mappings
        .keys()
        .map(Felt::from_bytes_le)
        .all(|key| key < Felt::from(BigInt::from(1).shl(128)));
    let number_of_keys = dict.mappings.len() as u64;

    // How we update the range check depends on whether we have any big key or not.
    // - If there are no big keys, every unique key increases the range check by 3.
    // - If there are big keys:
    //   - the first unique key increases the range check by 2.
    //   - the remaining unique keys increase the range check by 6.

    // The sierra-to-casm implementation calls the `SquashDict` after some initial validation.
    // https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/felt252_dict.rs?plain=1#L159
    //
    // For each unique key, the `SquashDictInner` function is called, which
    // loops over all accesses to that key. At the end, the function calls
    // itself recursively until all keys have been iterated.

    // If there are no big keys, the first range check usage is done by the
    // caller of the inner function, which implies that it appears in two places:
    // 1a. Once in `SquashDict`, right before calling the inner function for the first time.
    //     https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/felt252_dict.rs?plain=1#L326
    // 1b. Once at the end of `SquashDictInner`, right before recursing.
    //     https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/felt252_dict.rs?plain=1#L507
    if no_big_keys {
        *range_check_ptr += number_of_keys;
    }

    // The next two range check usages are done always inside of the inner
    // function (regardless of whether we have big keys or not).
    // 2.  https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/felt252_dict.rs?plain=1#L416
    // 3.  https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/felt252_dict.rs?plain=1#L480
    *range_check_ptr += 2 * number_of_keys;

    // If there are big keys, then we use the range check 4 additional times per key, except for the first key.
    // https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/felt252_dict.rs#L669-L674
    if !no_big_keys && number_of_keys > 1 {
        *range_check_ptr += 4 * (number_of_keys - 1);
    }

    // For each non unique accessed key, we increase the range check an additional time.
    // https://github.com/starkware-libs/cairo/blob/v2.12.0-dev.1/crates/cairo-lang-sierra-to-casm/src/invocations/felt252_dict.rs?plain=1#L602
    *range_check_ptr += dict.count.saturating_sub(dict.mappings.len() as u64);

    forget(dict);
}

/// Compute `ec_point_from_x_nz(x)` and store it.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_from_x_nz(
    point_ptr: &mut [[u8; 32]; 2],
) -> bool {
    point_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    let x = Felt::from_bytes_le(&point_ptr[0]);

    // https://github.com/starkware-libs/cairo/blob/aaad921bba52e729dc24ece07fab2edf09ccfa15/crates/cairo-lang-sierra-to-casm/src/invocations/ec.rs#L63

    let x2 = x * x;
    let x3 = x2 * x;
    let alpha_x_plus_beta = x + BETA;
    let rhs = x3 + alpha_x_plus_beta;
    // https://github.com/starkware-libs/cairo/blob/9b603b88c2e5a98eec1bb8f323260b7765e94911/crates/cairo-lang-runner/src/casm_run/mod.rs#L1825
    let y = rhs
        .sqrt()
        .unwrap_or_else(|| (Felt::THREE * rhs).sqrt().unwrap());
    let y = y.min(-y);

    match AffinePoint::new(x, y) {
        Ok(point) => {
            point_ptr[1] = point.y().to_bytes_le();
            true
        }
        Err(_) => false,
    }
}

/// Compute `ec_point_try_new_nz(x)`.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_try_new_nz(
    point_ptr: &mut [[u8; 32]; 2],
) -> bool {
    point_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    point_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    let x = Felt::from_bytes_le(&point_ptr[0]);
    let y = Felt::from_bytes_le(&point_ptr[1]);

    match AffinePoint::new(x, y) {
        Ok(point) => {
            point_ptr[0] = point.x().to_bytes_le();
            point_ptr[1] = point.y().to_bytes_le();
            true
        }
        Err(_) => false,
    }
}

/// Compute `ec_state_init()` and store the state back.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_init(state_ptr: &mut [[u8; 32]; 4]) {
    // https://github.com/starkware-libs/cairo/blob/aaad921bba52e729dc24ece07fab2edf09ccfa15/crates/cairo-lang-runner/src/casm_run/mod.rs#L1802
    let mut rng = rand::rng();
    let (random_x, random_y) = loop {
        // Randominzing 31 bytes to make sure is in range.
        let x_bytes: [u8; 31] = rng.random();
        let random_x = Felt::from_bytes_be_slice(&x_bytes);
        let random_y_squared = random_x * random_x * random_x + random_x + BETA;
        if let Some(random_y) = random_y_squared.sqrt() {
            break (random_x, random_y);
        }
    };

    // We already made sure its a valid point.
    let state = AffinePoint::new_unchecked(random_x, random_y);

    state_ptr[0] = state.x().to_bytes_le();
    state_ptr[1] = state.y().to_bytes_le();
    state_ptr[2] = state_ptr[0];
    state_ptr[3] = state_ptr[1];
}

/// Compute `ec_state_add(state, point)` and store the state back.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add(
    state_ptr: &mut [[u8; 32]; 4],
    point_ptr: &[[u8; 32]; 2],
) {
    state_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    state_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    let mut point_ptr = *point_ptr;
    point_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    point_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    // We use unchecked methods because the inputs must already be valid points.
    let mut state = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr[0]),
        Felt::from_bytes_le(&state_ptr[1]),
    );
    let point = AffinePoint::new_unchecked(
        Felt::from_bytes_le(&point_ptr[0]),
        Felt::from_bytes_le(&point_ptr[1]),
    );

    state += &point;
    let state = state.to_affine().unwrap();

    state_ptr[0] = state.x().to_bytes_le();
    state_ptr[1] = state.y().to_bytes_le();
}

/// Compute `ec_state_add_mul(state, scalar, point)` and store the state back.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add_mul(
    state_ptr: &mut [[u8; 32]; 4],
    scalar_ptr: &[u8; 32],
    point_ptr: &[[u8; 32]; 2],
) {
    state_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    state_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    let mut point_ptr = *point_ptr;
    point_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    point_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    let mut scalar_ptr = *scalar_ptr;
    scalar_ptr[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    // Here the points should already be checked as valid, so we can use unchecked.
    let mut state = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr[0]),
        Felt::from_bytes_le(&state_ptr[1]),
    );
    let point = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&point_ptr[0]),
        Felt::from_bytes_le(&point_ptr[1]),
    );
    let scalar = Felt::from_bytes_le(&scalar_ptr);

    state += &point.mul(scalar);
    let state = state.to_affine().unwrap();

    state_ptr[0] = state.x().to_bytes_le();
    state_ptr[1] = state.y().to_bytes_le();
}

/// Compute `ec_state_try_finalize_nz(state)` and store the result.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_try_finalize_nz(
    point_ptr: &mut [[u8; 32]; 2],
    state_ptr: &[[u8; 32]; 4],
) -> bool {
    let mut state_ptr = *state_ptr;
    state_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    state_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    state_ptr[2][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    state_ptr[3][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    // We use unchecked methods because the inputs must already be valid points.
    let state = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr[0]),
        Felt::from_bytes_le(&state_ptr[1]),
    );
    let random = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr[2]),
        Felt::from_bytes_le(&state_ptr[3]),
    );

    if state.x() == random.x() && state.y() == random.y() {
        false
    } else {
        let point = &state - &random;
        let point = point.to_affine().unwrap();

        point_ptr[0] = point.x().to_bytes_le();
        point_ptr[1] = point.y().to_bytes_le();

        true
    }
}

thread_local! {
    pub(crate) static BUILTIN_COSTS: Cell<BuiltinCosts> = const {
        // These default values shouldn't be accessible, they will be overriden before entering
        // compiled code.
        Cell::new(BuiltinCosts {
            r#const: 0,
            pedersen: 0,
            bitwise: 0,
            ecop: 0,
            poseidon: 0,
            add_mod: 0,
            mul_mod: 0,
        })
    };
}

/// Get the costs builtin from the internal thread local.
pub extern "C" fn cairo_native__get_costs_builtin() -> *const [u64; 7] {
    BUILTIN_COSTS.with(|x| x.as_ptr()) as *const [u64; 7]
}

// Utility methods for the print runtime function

/// Formats the given felts as a debug string.
fn format_for_debug(mut felts: IntoIter<Felt>) -> String {
    let mut items = Vec::new();
    while let Some(item) = format_next_item(&mut felts) {
        items.push(item);
    }
    if let [item] = &items[..] {
        if item.is_string {
            return item.item.clone();
        }
    }
    items
        .into_iter()
        .map(|item| {
            if item.is_string {
                format!("{}\n", item.item)
            } else {
                format!("[DEBUG]\t{}\n", item.item)
            }
        })
        .join("")
}

/// A formatted string representation of anything formattable (e.g. ByteArray, felt, short-string).
pub struct FormattedItem {
    /// The formatted string representing the item.
    item: String,
    /// Whether the item is a string.
    is_string: bool,
}
impl FormattedItem {
    /// Returns the formatted item as is.
    #[must_use]
    pub fn get(self) -> String {
        self.item
    }
    /// Wraps the formatted item with quote, if it's a string. Otherwise returns it as is.
    #[must_use]
    pub fn quote_if_string(self) -> String {
        if self.is_string {
            format!("\"{}\"", self.item)
        } else {
            self.item
        }
    }
}

pub const BYTE_ARRAY_MAGIC: &str =
    "46a6158a16a947e5916b2a2ca68501a45e93d7110e81aa2d6438b1c57c879a3";
pub const BYTES_IN_WORD: usize = 31;

/// Formats a string or a short string / `felt252`. Returns the formatted string and a boolean
/// indicating whether it's a string. If can't format the item, returns None.
pub fn format_next_item<T>(values: &mut T) -> Option<FormattedItem>
where
    T: Iterator<Item = Felt> + Clone,
{
    let first_felt = values.next()?;

    if first_felt == Felt::from_hex(BYTE_ARRAY_MAGIC).unwrap() {
        if let Some(string) = try_format_string(values) {
            return Some(FormattedItem {
                item: string,
                is_string: true,
            });
        }
    }
    Some(FormattedItem {
        item: format_short_string(&first_felt),
        is_string: false,
    })
}

/// Formats a `Felt252`, as a short string if possible.
fn format_short_string(value: &Felt) -> String {
    let hex_value = value.to_biguint();
    match as_cairo_short_string(value) {
        Some(as_string) => format!("{hex_value:#x} ('{as_string}')"),
        None => format!("{hex_value:#x}"),
    }
}

/// Tries to format a string, represented as a sequence of `Felt252`s.
/// If the sequence is not a valid serialization of a `ByteArray`, returns None and doesn't change the
/// given iterator (`values`).
fn try_format_string<T>(values: &mut T) -> Option<String>
where
    T: Iterator<Item = Felt> + Clone,
{
    // Clone the iterator and work with the clone. If the extraction of the string is successful,
    // change the original iterator to the one we worked with. If not, continue with the
    // original iterator at the original point.
    let mut cloned_values_iter = values.clone();

    let num_full_words = cloned_values_iter.next()?.to_usize()?;
    let full_words = cloned_values_iter
        .by_ref()
        .take(num_full_words)
        .collect_vec();
    let pending_word = cloned_values_iter.next()?;
    let pending_word_len = cloned_values_iter.next()?.to_usize()?;

    let full_words_string = full_words
        .into_iter()
        .map(|word| as_cairo_short_string_ex(&word, BYTES_IN_WORD))
        .collect::<Option<Vec<String>>>()?
        .join("");
    let pending_word_string = as_cairo_short_string_ex(&pending_word, pending_word_len)?;

    // Extraction was successful, change the original iterator to the one we worked with.
    *values = cloned_values_iter;

    Some(format!("{full_words_string}{pending_word_string}"))
}

/// Converts a bigint representing a felt252 to a Cairo short-string.
#[must_use]
pub fn as_cairo_short_string(value: &Felt) -> Option<String> {
    let mut as_string = String::default();
    let mut is_end = false;
    for byte in value.to_biguint().to_bytes_be() {
        if byte == 0 {
            is_end = true;
        } else if is_end {
            return None;
        } else if byte.is_ascii_graphic() || byte.is_ascii_whitespace() {
            as_string.push(byte as char);
        } else {
            return None;
        }
    }
    Some(as_string)
}

/// Converts a bigint representing a felt252 to a Cairo short-string of the given length.
/// Nulls are allowed and length must be <= 31.
#[must_use]
pub fn as_cairo_short_string_ex(value: &Felt, length: usize) -> Option<String> {
    if length == 0 {
        return if value.is_zero() {
            Some(String::new())
        } else {
            None
        };
    }
    if length > 31 {
        // A short string can't be longer than 31 bytes.
        return None;
    }

    // We pass through biguint as felt252.to_bytes_be() does not trim leading zeros.
    let bytes = value.to_biguint().to_bytes_be();
    let bytes_len = bytes.len();
    if bytes_len > length {
        // `value` has more bytes than expected.
        return None;
    }

    let mut as_string = String::new();
    for byte in bytes {
        if byte == 0 {
            as_string.push_str(r"\0");
        } else if byte.is_ascii_graphic() || byte.is_ascii_whitespace() {
            as_string.push(byte as char);
        } else {
            as_string.push_str(format!(r"\x{:02x}", byte).as_str());
        }
    }

    // `to_bytes_be` misses starting nulls. Prepend them as needed.
    let missing_nulls = length - bytes_len;
    as_string.insert_str(0, &r"\0".repeat(missing_nulls));

    Some(as_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env, fs,
        io::{Read, Seek},
        os::fd::AsRawFd,
    };

    pub fn felt252_short_str(value: &str) -> Felt {
        let values: Vec<_> = value
            .chars()
            .filter_map(|c| c.is_ascii().then_some(c as u8))
            .collect();

        assert!(values.len() < 32);
        Felt::from_bytes_be_slice(&values)
    }

    #[test]
    fn test_debug_print() {
        let dir = env::temp_dir();
        fs::remove_file(dir.join("print.txt")).ok();
        let mut file = File::create_new(dir.join("print.txt")).unwrap();
        {
            let fd = file.as_raw_fd();
            let data = felt252_short_str("hello world");
            let data = data.to_bytes_le();
            unsafe { cairo_native__libfunc__debug__print(fd, &data, 1) };
        }
        file.seek(std::io::SeekFrom::Start(0)).unwrap();

        let mut result = String::new();
        file.read_to_string(&mut result).unwrap();

        assert_eq!(
            result,
            "[DEBUG]\t0x68656c6c6f20776f726c64 ('hello world')\n"
        );
    }

    #[test]
    fn test_pederesen() {
        let mut dst = [0; 32];
        let lhs = Felt::from(1).to_bytes_le();
        let rhs = Felt::from(3).to_bytes_le();

        unsafe {
            cairo_native__libfunc__pedersen(&mut dst, &lhs, &rhs);
        }

        assert_eq!(
            dst,
            [
                84, 98, 174, 134, 3, 124, 237, 179, 166, 110, 159, 98, 170, 35, 83, 237, 130, 154,
                236, 0, 205, 134, 200, 185, 39, 92, 0, 228, 132, 217, 130, 5
            ]
        )
    }

    #[test]
    fn test_hades_permutation() {
        let mut op0 = Felt::from(1).to_bytes_le();
        let mut op1 = Felt::from(1).to_bytes_le();
        let mut op2 = Felt::from(1).to_bytes_le();

        unsafe {
            cairo_native__libfunc__hades_permutation(&mut op0, &mut op1, &mut op2);
        }

        assert_eq!(
            Felt::from_bytes_le(&op0),
            Felt::from_hex("0x4ebdde1149fcacbb41e4fc342432a48c97994fd045f432ad234ae9279269779")
                .unwrap()
        );
        assert_eq!(
            Felt::from_bytes_le(&op1),
            Felt::from_hex("0x7f4cec57dd08b69414f7de7dffa230fc90fa3993673c422408af05831e0cc98")
                .unwrap()
        );
        assert_eq!(
            Felt::from_bytes_le(&op2),
            Felt::from_hex("0x5b5d00fd09caade43caffe70527fa84d5d9cd51e22c2ce115693ecbb5854d6a")
                .unwrap()
        );
    }

    #[test]
    fn test_dict() {
        let dict = unsafe {
            cairo_native__dict_new(size_of::<u64>() as u64, align_of::<u64>() as u64, None)
        };

        let key = Felt::ONE.to_bytes_le();
        let mut ptr = ptr::null_mut::<u64>();

        assert_eq!(
            unsafe { cairo_native__dict_get(dict, &key, (&raw mut ptr).cast()) },
            0,
        );
        assert!(!ptr.is_null());
        unsafe { *ptr = 24 };

        assert_eq!(
            unsafe { cairo_native__dict_get(dict, &key, (&raw mut ptr).cast()) },
            1,
        );
        assert!(!ptr.is_null());
        assert_eq!(unsafe { *ptr }, 24);
        unsafe { *ptr = 42 };

        let mut range_check = 0;
        let mut gas = 0;

        unsafe { cairo_native__dict_squash(dict, &mut range_check, &mut gas) };
        assert_eq!(gas, 4050);

        let cloned_dict = unsafe { cairo_native__dict_dup(dict) };
        unsafe { cairo_native__dict_drop(dict) };

        assert_eq!(
            unsafe { cairo_native__dict_get(cloned_dict, &key, (&raw mut ptr).cast()) },
            1,
        );
        assert!(!ptr.is_null());
        assert_eq!(unsafe { *ptr }, 42);

        unsafe { cairo_native__dict_drop(cloned_dict) };
    }

    #[test]
    fn test_ec__ec_point() {
        let mut state = [
            Felt::ZERO.to_bytes_le(),
            Felt::ZERO.to_bytes_le(),
            Felt::ZERO.to_bytes_le(),
            Felt::ZERO.to_bytes_le(),
        ];

        unsafe { cairo_native__libfunc__ec__ec_state_init(&mut state) };

        let points: &mut [[u8; 32]; 2] = (&mut state[..2]).try_into().unwrap();

        let result = unsafe { cairo_native__libfunc__ec__ec_point_try_new_nz(points) };

        // point should be valid since it was made with state init
        assert!(result);
    }

    #[test]
    fn test_ec__ec_point_add() {
        // Test values taken from starknet-rs
        let mut state = [
            Felt::from_dec_str(
                "874739451078007766457464989774322083649278607533249481151382481072868806602",
            )
            .unwrap()
            .to_bytes_le(),
            Felt::from_dec_str(
                "152666792071518830868575557812948353041420400780739481342941381225525861407",
            )
            .unwrap()
            .to_bytes_le(),
            Felt::from_dec_str(
                "874739451078007766457464989774322083649278607533249481151382481072868806602",
            )
            .unwrap()
            .to_bytes_le(),
            Felt::from_dec_str(
                "152666792071518830868575557812948353041420400780739481342941381225525861407",
            )
            .unwrap()
            .to_bytes_le(),
        ];

        let point = [
            Felt::from_dec_str(
                "874739451078007766457464989774322083649278607533249481151382481072868806602",
            )
            .unwrap()
            .to_bytes_le(),
            Felt::from_dec_str(
                "152666792071518830868575557812948353041420400780739481342941381225525861407",
            )
            .unwrap()
            .to_bytes_le(),
        ];

        unsafe {
            cairo_native__libfunc__ec__ec_state_add(&mut state, &point);
        };

        assert_eq!(
            state[0],
            Felt::from_dec_str(
                "3324833730090626974525872402899302150520188025637965566623476530814354734325",
            )
            .unwrap()
            .to_bytes_le()
        );
        assert_eq!(
            state[1],
            Felt::from_dec_str(
                "3147007486456030910661996439995670279305852583596209647900952752170983517249",
            )
            .unwrap()
            .to_bytes_le()
        );
    }
}
