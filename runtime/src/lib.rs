#![allow(non_snake_case)]

use cairo_lang_sierra_gas::core_libfunc_cost::{
    DICT_SQUASH_REPEATED_ACCESS_COST, DICT_SQUASH_UNIQUE_KEY_COST,
};
use itertools::Itertools;
use lazy_static::lazy_static;
use num_traits::{ToPrimitive, Zero};
use rand::Rng;
use starknet_curve::curve_params::BETA;
use starknet_types_core::{
    curve::{AffinePoint, ProjectivePoint},
    felt::Felt,
    hash::StarkHash,
};
use std::{collections::HashMap, fs::File, io::Write, os::fd::FromRawFd, ptr::NonNull, slice};
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__debug__print(
    target_fd: i32,
    data: *const [u8; 32],
    len: u32,
) -> i32 {
    let mut target = File::from_raw_fd(target_fd);

    let mut items = Vec::with_capacity(len as usize);

    for i in 0..len as usize {
        let data = *data.add(i);

        let value = Felt::from_bytes_le(&data);
        items.push(value);
    }

    let value = format_for_debug(items.into_iter());

    if write!(target, "{}", value).is_err() {
        return 1;
    };

    // Avoid closing `stdout`.
    std::mem::forget(target);

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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__pedersen(
    dst: *mut u8,
    lhs: *const u8,
    rhs: *const u8,
) {
    // Extract arrays from the pointers.
    let dst = slice::from_raw_parts_mut(dst, 32);
    let lhs = slice::from_raw_parts(lhs, 32);
    let rhs = slice::from_raw_parts(rhs, 32);

    // Convert to FieldElement.
    let lhs = Felt::from_bytes_le_slice(lhs);
    let rhs = Felt::from_bytes_le_slice(rhs);

    // Compute pedersen hash and copy the result into `dst`.
    let res = starknet_types_core::hash::Pedersen::hash(&lhs, &rhs);
    dst.copy_from_slice(&res.to_bytes_le());
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__hades_permutation(
    op0: *mut u8,
    op1: *mut u8,
    op2: *mut u8,
) {
    // Extract arrays from the pointers.
    let op0 = slice::from_raw_parts_mut(op0, 32);
    let op1 = slice::from_raw_parts_mut(op1, 32);
    let op2 = slice::from_raw_parts_mut(op2, 32);

    // Convert to FieldElement.
    let mut state = [
        Felt::from_bytes_le_slice(op0),
        Felt::from_bytes_le_slice(op1),
        Felt::from_bytes_le_slice(op2),
    ];

    // Compute Poseidon permutation.
    starknet_types_core::hash::Poseidon::hades_permutation(&mut state);

    // Write back the results.
    op0.copy_from_slice(&state[0].to_bytes_le());
    op1.copy_from_slice(&state[1].to_bytes_le());
    op2.copy_from_slice(&state[2].to_bytes_le());
}

/// Felt252 type used in cairo native runtime
pub type FeltDict = (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64);

/// Allocates a new dictionary. Internally a rust hashmap: `HashMap<[u8; 32], NonNull<()>`
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__alloc_dict() -> *mut std::ffi::c_void {
    Box::into_raw(Box::<FeltDict>::default()) as _
}

/// Frees the dictionary.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_free(ptr: *mut FeltDict) {
    let mut map = Box::from_raw(ptr);

    // Free the entries manually.
    for (_, entry) in map.as_mut().0.drain() {
        libc::free(entry.as_ptr().cast());
    }
}

/// Needed for the correct alignment,
/// since the key [u8; 32] in rust has 8 byte alignment but its a felt,
/// so in reality it has 16.
#[repr(C, align(16))]
pub struct DictValuesArrayAbi {
    pub key: [u8; 32],
    pub value: std::ptr::NonNull<libc::c_void>,
}

/// Returns a array over the values of the dict, used for deep cloning.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_values(
    ptr: *mut FeltDict,
    len: *mut u64,
) -> *mut DictValuesArrayAbi {
    let dict: &mut FeltDict = &mut *ptr;

    let values: Vec<_> = dict
        .0
        .clone()
        .into_iter()
        // make it ffi safe for use within MLIR.
        .map(|x| DictValuesArrayAbi {
            key: x.0,
            value: x.1,
        })
        .collect();
    *len = values.len() as u64;
    values.leak::<'static>().as_mut_ptr()
}

/// Gets the value for a given key, the returned pointer is null if not found.
/// Increments the access count.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_get(
    ptr: *mut FeltDict,
    key: &[u8; 32],
) -> *mut std::ffi::c_void {
    let dict: &mut FeltDict = &mut *ptr;
    let map = &dict.0;
    dict.1 += 1;

    if let Some(v) = map.get(key) {
        v.as_ptr()
    } else {
        std::ptr::null_mut()
    }
}

/// Inserts the provided key value. Returning the old one or nullptr if there was none.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_insert(
    ptr: *mut FeltDict,
    key: &[u8; 32],
    value: NonNull<std::ffi::c_void>,
) -> *mut std::ffi::c_void {
    let dict = &mut *ptr;
    let old_ptr = dict.0.insert(*key, value);

    if let Some(v) = old_ptr {
        v.as_ptr()
    } else {
        std::ptr::null_mut()
    }
}

/// Compute the total gas refund for the dictionary at squash time.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_gas_refund(ptr: *const FeltDict) -> u64 {
    let dict = &*ptr;
    (dict.1 - dict.0.len() as u64) * *DICT_GAS_REFUND_PER_ACCESS
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_from_x_nz(
    mut point_ptr: NonNull<[[u8; 32]; 2]>,
) -> bool {
    let x = Felt::from_bytes_le(&point_ptr.as_ref()[0]);

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
            point_ptr.as_mut()[1].copy_from_slice(&point.y().to_bytes_le());
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_try_new_nz(
    mut point_ptr: NonNull<[[u8; 32]; 2]>,
) -> bool {
    let x = Felt::from_bytes_le(&point_ptr.as_ref()[0]);
    let y = Felt::from_bytes_le(&point_ptr.as_ref()[1]);

    match AffinePoint::new(x, y) {
        Ok(point) => {
            point_ptr.as_mut()[0].copy_from_slice(&point.x().to_bytes_le());
            point_ptr.as_mut()[1].copy_from_slice(&point.y().to_bytes_le());
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_init(
    mut state_ptr: NonNull<[[u8; 32]; 4]>,
) {
    // https://github.com/starkware-libs/cairo/blob/aaad921bba52e729dc24ece07fab2edf09ccfa15/crates/cairo-lang-runner/src/casm_run/mod.rs#L1802
    let mut rng = rand::thread_rng();
    let (random_x, random_y) = loop {
        // Randominzing 31 bytes to make sure is in range.
        let x_bytes: [u8; 31] = rng.gen();
        let random_x = Felt::from_bytes_be_slice(&x_bytes);
        let random_y_squared = random_x * random_x * random_x + random_x + BETA;
        if let Some(random_y) = random_y_squared.sqrt() {
            break (random_x, random_y);
        }
    };

    // We already made sure its a valid point.
    let state = AffinePoint::new_unchecked(random_x, random_y);

    state_ptr.as_mut()[0].copy_from_slice(&state.x().to_bytes_le());
    state_ptr.as_mut()[1].copy_from_slice(&state.y().to_bytes_le());
    state_ptr.as_mut()[2].copy_from_slice(&state.x().to_bytes_le());
    state_ptr.as_mut()[3].copy_from_slice(&state.y().to_bytes_le());
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add(
    mut state_ptr: NonNull<[[u8; 32]; 4]>,
    point_ptr: NonNull<[[u8; 32]; 2]>,
) {
    // We use unchecked methods because the inputs must already be valid points.
    let mut state = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr.as_ref()[0]),
        Felt::from_bytes_le(&state_ptr.as_ref()[1]),
    );
    let point = AffinePoint::new_unchecked(
        Felt::from_bytes_le(&point_ptr.as_ref()[0]),
        Felt::from_bytes_le(&point_ptr.as_ref()[1]),
    );

    state += &point;
    let state = state.to_affine().unwrap();

    state_ptr.as_mut()[0].copy_from_slice(&state.x().to_bytes_le());
    state_ptr.as_mut()[1].copy_from_slice(&state.y().to_bytes_le());
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add_mul(
    mut state_ptr: NonNull<[[u8; 32]; 4]>,
    scalar_ptr: NonNull<[u8; 32]>,
    point_ptr: NonNull<[[u8; 32]; 2]>,
) {
    // Here the points should already be checked as valid, so we can use unchecked.
    let mut state = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr.as_ref()[0]),
        Felt::from_bytes_le(&state_ptr.as_ref()[1]),
    );
    let point = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&point_ptr.as_ref()[0]),
        Felt::from_bytes_le(&point_ptr.as_ref()[1]),
    );
    let scalar = Felt::from_bytes_le(scalar_ptr.as_ref());

    state += &point.mul(scalar);
    let state = state.to_affine().unwrap();

    state_ptr.as_mut()[0].copy_from_slice(&state.x().to_bytes_le());
    state_ptr.as_mut()[1].copy_from_slice(&state.y().to_bytes_le());
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_try_finalize_nz(
    mut point_ptr: NonNull<[[u8; 32]; 2]>,
    state_ptr: NonNull<[[u8; 32]; 4]>,
) -> bool {
    // We use unchecked methods because the inputs must already be valid points.
    let state = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr.as_ref()[0]),
        Felt::from_bytes_le(&state_ptr.as_ref()[1]),
    );
    let random = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr.as_ref()[2]),
        Felt::from_bytes_le(&state_ptr.as_ref()[3]),
    );

    if state.x() == random.x() && state.y() == random.y() {
        false
    } else {
        let point = &state - &random;
        let point = point.to_affine().unwrap();

        point_ptr.as_mut()[0].copy_from_slice(&point.x().to_bytes_le());
        point_ptr.as_mut()[1].copy_from_slice(&point.y().to_bytes_le());

        true
    }
}

/// Utility methods for the print runtime function

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
    pub fn get(self) -> String {
        self.item
    }
    /// Wraps the formatted item with quote, if it's a string. Otherwise returns it as is.
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
/// If the sequence is not a valid serialization of a ByteArray, returns None and doesn't change the
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
pub fn as_cairo_short_string_ex(value: &Felt, length: usize) -> Option<String> {
    if length == 0 {
        return if value.is_zero() {
            Some("".to_string())
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

    let mut as_string = "".to_string();
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
