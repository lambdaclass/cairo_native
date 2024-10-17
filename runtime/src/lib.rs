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
use std::{collections::HashMap, ffi::c_void, fs::File, io::Write, os::fd::FromRawFd};
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
        let mut data = *data.add(i);
        data[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

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
#[no_mangle]
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
    pub inner: HashMap<[u8; 32], *mut c_void>,
    pub count: u64,

    pub free_fn: unsafe extern "C" fn(*mut c_void),
}

/// Allocate a new dictionary.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_new(
    free_fn: extern "C" fn(*mut c_void),
) -> *mut FeltDict {
    Box::into_raw(Box::new(FeltDict {
        inner: HashMap::default(),
        count: 0,
        free_fn,
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_drop(
    ptr: *mut FeltDict,
    drop_fn: Option<extern "C" fn(*mut c_void)>,
) {
    let dict = Box::from_raw(ptr);

    // Free the entries manually.
    for entry in dict.inner.into_values() {
        if !entry.is_null() {
            if let Some(drop_fn) = drop_fn {
                drop_fn(entry);
            }

            (dict.free_fn)(entry);
        }
    }
}

/// Duplicate a dictionary using a provided callback to clone each element.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_dup(
    ptr: *mut FeltDict,
    dup_fn: extern "C" fn(*mut c_void) -> *mut c_void,
) -> *mut FeltDict {
    let old_dict = &*ptr;
    let mut new_dict = Box::new(FeltDict {
        inner: HashMap::default(),
        count: 0,
        free_fn: old_dict.free_fn,
    });

    new_dict.inner.extend(
        old_dict
            .inner
            .iter()
            .filter_map(|(&k, &v)| (!v.is_null()).then_some((k, dup_fn(v)))),
    );

    Box::into_raw(new_dict)
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__dict_get(
    dict: &mut FeltDict,
    key: &[u8; 32],
) -> *mut c_void {
    let mut key = *key;
    key[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    dict.count += 1;
    dict.inner.entry(key).or_insert(std::ptr::null_mut()) as *mut _ as *mut c_void
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
    (dict.count - dict.inner.len() as u64) * *DICT_GAS_REFUND_PER_ACCESS
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
#[no_mangle]
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
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_init(state_ptr: &mut [[u8; 32]; 4]) {
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
#[no_mangle]
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
#[no_mangle]
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
#[no_mangle]
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

#[cfg(test)]
mod tests {
    use std::{
        env,
        fs::{remove_file, File},
        io::{Read, Seek},
        os::{fd::AsRawFd, raw::c_void},
    };

    use starknet_types_core::felt::Felt;

    use crate::{
        cairo_native__dict_new, cairo_native__libfunc__debug__print,
        cairo_native__libfunc__hades_permutation, cairo_native__libfunc__pedersen,
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
        remove_file(dir.join("print.txt")).ok();
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

    // Test free_fn for the dict testing, values are u64.
    pub extern "C" fn free_fn_test(ptr: *mut c_void) {
        assert!(!ptr.is_null());
        let b: Box<u64> = unsafe { Box::from_raw(ptr.cast()) };
        drop(b);
    }

    #[test]
    fn test_dict() {
        let dict = unsafe { cairo_native__dict_new(free_fn_test) };
    }
}
