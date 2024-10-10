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
use std::{collections::HashMap, ffi::c_void, fs::File, io::Write, os::fd::FromRawFd, slice};
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
    dict.count += 1;
    dict.inner.entry(*key).or_insert(std::ptr::null_mut()) as *mut _ as *mut c_void
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
    point_ptr: &mut [[u8; 32]; 2],
) -> bool {
    let x = Felt::from_bytes_le(&point_ptr[0]);
    let y = Felt::from_bytes_le(&point_ptr[1]);

    match AffinePoint::new(x, y) {
        Ok(point) => {
            point_ptr[0].copy_from_slice(&point.x().to_bytes_le());
            point_ptr[1].copy_from_slice(&point.y().to_bytes_le());
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

    state_ptr[0].copy_from_slice(&state.x().to_bytes_le());
    state_ptr[1].copy_from_slice(&state.y().to_bytes_le());
    state_ptr[2].copy_from_slice(&state.x().to_bytes_le());
    state_ptr[3].copy_from_slice(&state.y().to_bytes_le());
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

    state_ptr[0].copy_from_slice(&state.x().to_bytes_le());
    state_ptr[1].copy_from_slice(&state.y().to_bytes_le());
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
    // Here the points should already be checked as valid, so we can use unchecked.
    let mut state = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr[0]),
        Felt::from_bytes_le(&state_ptr[1]),
    );
    let point = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&point_ptr[0]),
        Felt::from_bytes_le(&point_ptr[1]),
    );
    let scalar = Felt::from_bytes_le(scalar_ptr);

    state += &point.mul(scalar);
    let state = state.to_affine().unwrap();

    state_ptr[0].copy_from_slice(&state.x().to_bytes_le());
    state_ptr[1].copy_from_slice(&state.y().to_bytes_le());
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

        point_ptr[0].copy_from_slice(&point.x().to_bytes_le());
        point_ptr[1].copy_from_slice(&point.y().to_bytes_le());

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

#[cfg(feature = "with-trace-dump")]
pub mod trace_dump {
    use cairo_lang_sierra::{
        extensions::{
            bounded_int::BoundedIntConcreteType,
            core::{CoreLibfunc, CoreType, CoreTypeConcrete},
            starknet::StarkNetTypeConcrete,
        },
        ids::{ConcreteTypeId, VarId},
        program::StatementIdx,
        program_registry::ProgramRegistry,
    };
    use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
    use num_bigint::BigInt;
    use num_traits::One;
    use sierra_emu::{ProgramTrace, StateDump, Value};
    use starknet_types_core::felt::Felt;
    use std::{
        alloc::Layout,
        collections::HashMap,
        mem::swap,
        ops::Range,
        ptr::NonNull,
        sync::{LazyLock, Mutex},
    };

    use crate::FeltDict;

    pub static TRACE_DUMP: LazyLock<Mutex<HashMap<u64, TraceDump>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    pub struct TraceDump {
        pub trace: ProgramTrace,
        state: OrderedHashMap<VarId, Value>,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,

        get_layout: fn(&CoreTypeConcrete, &ProgramRegistry<CoreType, CoreLibfunc>) -> Layout,
    }

    impl TraceDump {
        pub fn new(
            registry: ProgramRegistry<CoreType, CoreLibfunc>,
            get_layout: fn(&CoreTypeConcrete, &ProgramRegistry<CoreType, CoreLibfunc>) -> Layout,
        ) -> Self {
            Self {
                trace: ProgramTrace::default(),
                state: OrderedHashMap::default(),
                registry,

                get_layout,
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn get_trace_dump_ptr() -> *const Mutex<HashMap<u64, TraceDump>> {
        &*TRACE_DUMP as *const _
    }

    #[no_mangle]
    pub unsafe extern "C" fn cairo_native__trace_dump__state(
        trace_id: u64,
        var_id: u64,
        type_id: u64,
        value_ptr: NonNull<()>,
    ) {
        let mut trace_dump = TRACE_DUMP.lock().unwrap();
        let trace_dump = trace_dump.get_mut(&trace_id).unwrap();

        let type_id = ConcreteTypeId::new(type_id);
        let value = read_value_ptr(
            &trace_dump.registry,
            &type_id,
            value_ptr,
            trace_dump.get_layout,
        );

        trace_dump.state.insert(VarId::new(var_id), value);
    }

    #[no_mangle]
    pub unsafe extern "C" fn cairo_native__trace_dump__push(trace_id: u64, statement_idx: u64) {
        let mut trace_dump = TRACE_DUMP.lock().unwrap();
        let trace_dump = trace_dump.get_mut(&trace_id).unwrap();

        let mut items = OrderedHashMap::default();
        swap(&mut items, &mut trace_dump.state);

        trace_dump
            .trace
            .push(StateDump::new(StatementIdx(statement_idx as usize), items));
    }

    unsafe fn read_value_ptr(
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        type_id: &ConcreteTypeId,
        value_ptr: NonNull<()>,
        get_layout: fn(&CoreTypeConcrete, &ProgramRegistry<CoreType, CoreLibfunc>) -> Layout,
    ) -> Value {
        let type_info = registry.get_type(type_id).unwrap();
        match type_info {
            CoreTypeConcrete::Felt252(_)
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::ContractAddress(_))
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::ClassHash(_))
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::StorageAddress(_))
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::StorageBaseAddress(_)) => {
                Value::Felt(Felt::from_bytes_le(value_ptr.cast().as_ref()))
            }
            CoreTypeConcrete::Uint8(_) => Value::U8(value_ptr.cast().read()),
            CoreTypeConcrete::Uint16(_) => Value::U16(value_ptr.cast().read()),
            CoreTypeConcrete::Uint32(_) => Value::U32(value_ptr.cast().read()),
            CoreTypeConcrete::Uint64(_) => Value::U64(value_ptr.cast().read()),
            CoreTypeConcrete::Uint128(_) | CoreTypeConcrete::GasBuiltin(_) => {
                Value::U128(value_ptr.cast().read())
            }

            CoreTypeConcrete::BoundedInt(BoundedIntConcreteType { range, .. }) => {
                let n_bits = ((range.size() - BigInt::one()).bits() as u32).max(1);
                let n_bytes = n_bits.next_multiple_of(8) >> 3;

                let data = NonNull::slice_from_raw_parts(value_ptr.cast::<u8>(), n_bytes as usize);

                let value = BigInt::from_bytes_le(num_bigint::Sign::Plus, data.as_ref());

                Value::BoundedInt {
                    range: Range {
                        start: range.lower.clone(),
                        end: range.upper.clone(),
                    },
                    value: value + &range.lower,
                }
            }

            CoreTypeConcrete::EcPoint(_) => {
                let layout = Layout::new::<()>();
                let (x, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (y, _) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };

                Value::EcPoint { x, y }
            }
            CoreTypeConcrete::EcState(_) => {
                let layout = Layout::new::<()>();
                let (x0, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (y0, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (x1, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (y1, _) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };

                Value::EcState { x0, y0, x1, y1 }
            }

            CoreTypeConcrete::Uninitialized(info) => Value::Uninitialized {
                ty: info.ty.clone(),
            },
            CoreTypeConcrete::Box(info) => read_value_ptr(
                registry,
                &info.ty,
                value_ptr.cast::<NonNull<()>>().read(),
                get_layout,
            ),
            CoreTypeConcrete::Array(info) => {
                let layout = Layout::new::<()>();
                let (array_ptr, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<*mut ()>()).unwrap();
                    (value_ptr.byte_add(offset).cast::<*mut ()>().read(), layout)
                };
                let (array_begin, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<u32>()).unwrap();
                    (value_ptr.byte_add(offset).cast::<u32>().read(), layout)
                };
                let (array_end, _) = {
                    let (layout, offset) = layout.extend(Layout::new::<u32>()).unwrap();
                    (value_ptr.byte_add(offset).cast::<u32>().read(), layout)
                };

                let layout =
                    get_layout(registry.get_type(&info.ty).unwrap(), registry).pad_to_align();

                let mut data = Vec::with_capacity((array_end - array_begin) as usize);
                for index in array_begin..array_end {
                    let index = index as usize;

                    data.push(read_value_ptr(
                        registry,
                        &info.ty,
                        NonNull::new(array_ptr.byte_add(layout.size() * index)).unwrap(),
                        get_layout,
                    ));
                }

                Value::Array {
                    ty: info.ty.clone(),
                    data,
                }
            }

            CoreTypeConcrete::Struct(info) => {
                let mut layout = Layout::new::<()>();
                let mut members = Vec::with_capacity(info.members.len());
                for member_ty in &info.members {
                    let type_info = registry.get_type(member_ty).unwrap();
                    let member_layout = get_layout(type_info, registry);

                    let offset;
                    (layout, offset) = layout.extend(member_layout).unwrap();

                    let current_ptr = value_ptr.byte_add(offset);
                    members.push(read_value_ptr(registry, member_ty, current_ptr, get_layout));
                }

                Value::Struct(members)
            }
            CoreTypeConcrete::Enum(info) => {
                let tag_bits = info.variants.len().next_power_of_two().trailing_zeros();
                let (tag_value, layout) = match tag_bits {
                    0 => todo!(),
                    width if width <= 8 => {
                        (value_ptr.cast::<u8>().read() as usize, Layout::new::<u8>())
                    }
                    width if width <= 16 => (
                        value_ptr.cast::<u16>().read() as usize,
                        Layout::new::<u16>(),
                    ),
                    width if width <= 32 => (
                        value_ptr.cast::<u32>().read() as usize,
                        Layout::new::<u32>(),
                    ),
                    width if width <= 64 => (
                        value_ptr.cast::<u64>().read() as usize,
                        Layout::new::<u64>(),
                    ),
                    width if width <= 128 => (
                        value_ptr.cast::<u128>().read() as usize,
                        Layout::new::<u128>(),
                    ),
                    _ => todo!(),
                };

                let payload = {
                    let (_, offset) = layout
                        .extend(get_layout(
                            registry.get_type(&info.variants[tag_value]).unwrap(),
                            registry,
                        ))
                        .unwrap();

                    read_value_ptr(
                        registry,
                        &info.variants[tag_value],
                        value_ptr.byte_add(offset),
                        get_layout,
                    )
                };

                Value::Enum {
                    self_ty: type_id.clone(),
                    index: tag_value,
                    payload: Box::new(payload),
                }
            }

            CoreTypeConcrete::NonZero(info) | CoreTypeConcrete::Snapshot(info) => {
                read_value_ptr(registry, &info.ty, value_ptr, get_layout)
            }

            // Builtins and other unit types:
            CoreTypeConcrete::Bitwise(_)
            | CoreTypeConcrete::BuiltinCosts(_)
            | CoreTypeConcrete::EcOp(_)
            | CoreTypeConcrete::Pedersen(_)
            | CoreTypeConcrete::Poseidon(_)
            | CoreTypeConcrete::RangeCheck(_)
            | CoreTypeConcrete::SegmentArena(_)
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_))
            | CoreTypeConcrete::Uint128MulGuarantee(_) => Value::Unit,

            // TODO:
            CoreTypeConcrete::Coupon(_) => todo!("CoreTypeConcrete::Coupon"),
            CoreTypeConcrete::Circuit(_) => todo!("CoreTypeConcrete::Circuit"),
            CoreTypeConcrete::Const(_) => todo!("CoreTypeConcrete::Const"),
            CoreTypeConcrete::Sint8(_) => Value::I8(value_ptr.cast().read()),
            CoreTypeConcrete::Sint16(_) => todo!("CoreTypeConcrete::Sint16"),
            CoreTypeConcrete::Sint32(_) => Value::I32(value_ptr.cast().read()),
            CoreTypeConcrete::Sint64(_) => todo!("CoreTypeConcrete::Sint64"),
            CoreTypeConcrete::Sint128(_) => Value::I128(value_ptr.cast().read()),
            CoreTypeConcrete::Nullable(info) => {
                let inner_ptr = value_ptr.cast::<*mut ()>().read();
                match NonNull::new(inner_ptr) {
                    Some(inner_ptr) => read_value_ptr(registry, &info.ty, inner_ptr, get_layout),
                    None => Value::Uninitialized {
                        ty: info.ty.clone(),
                    },
                }
            }

            CoreTypeConcrete::RangeCheck96(_) => todo!("CoreTypeConcrete::RangeCheck96"),
            CoreTypeConcrete::Felt252Dict(info) => {
                let value = value_ptr.cast::<FeltDict>().as_ref();

                let data = value
                    .inner
                    .iter()
                    .map(|(k, &p)| {
                        let v = match NonNull::new(p) {
                            Some(value_ptr) => {
                                read_value_ptr(registry, &info.ty, value_ptr.cast(), get_layout)
                            }
                            None => Value::Uninitialized {
                                ty: info.ty.clone(),
                            },
                        };
                        let k = Felt::from_bytes_le(k);
                        (k, v)
                    })
                    .collect::<HashMap<Felt, Value>>();

                Value::FeltDict {
                    ty: info.ty.clone(),
                    data,
                }
            }
            CoreTypeConcrete::Felt252DictEntry(_) => todo!("CoreTypeConcrete::Felt252DictEntry"),
            CoreTypeConcrete::SquashedFelt252Dict(_) => {
                todo!("CoreTypeConcrete::SquashedFelt252Dict")
            }
            CoreTypeConcrete::Span(_) => todo!("CoreTypeConcrete::Span"),
            CoreTypeConcrete::StarkNet(selector) => match selector {
                StarkNetTypeConcrete::Secp256Point(_) => {
                    todo!("StarkNetTypeConcrete::Secp256Point")
                }
                StarkNetTypeConcrete::Sha256StateHandle(_) => {
                    todo!("StarkNetTypeConcrete::Sha256StateHandle")
                }
                _ => unreachable!(),
            },
            CoreTypeConcrete::Bytes31(_) => todo!("CoreTypeConcrete::Bytes31"),
        }
    }
}
