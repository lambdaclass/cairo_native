//#![allow(non_snake_case)]
#![allow(non_snake_case)]
//

//use cairo_lang_sierra_gas::core_libfunc_cost::{
use cairo_lang_sierra_gas::core_libfunc_cost::{
//    DICT_SQUASH_REPEATED_ACCESS_COST, DICT_SQUASH_UNIQUE_KEY_COST,
    DICT_SQUASH_REPEATED_ACCESS_COST, DICT_SQUASH_UNIQUE_KEY_COST,
//};
};
//use lazy_static::lazy_static;
use lazy_static::lazy_static;
//use starknet_crypto::FieldElement;
use starknet_crypto::FieldElement;
//use starknet_curve::AffinePoint;
use starknet_curve::AffinePoint;
//use starknet_types_core::felt::Felt;
use starknet_types_core::felt::Felt;
//use std::{collections::HashMap, fs::File, io::Write, os::fd::FromRawFd, ptr::NonNull, slice};
use std::{collections::HashMap, fs::File, io::Write, os::fd::FromRawFd, ptr::NonNull, slice};
//

//lazy_static! {
lazy_static! {
//    pub static ref HALF_PRIME: FieldElement = FieldElement::from_dec_str(
    pub static ref HALF_PRIME: FieldElement = FieldElement::from_dec_str(
//        "1809251394333065606848661391547535052811553607665798349986546028067936010240"
        "1809251394333065606848661391547535052811553607665798349986546028067936010240"
//    )
    )
//    .unwrap();
    .unwrap();
//    pub static ref DICT_GAS_REFUND_PER_ACCESS: u64 =
    pub static ref DICT_GAS_REFUND_PER_ACCESS: u64 =
//        (DICT_SQUASH_UNIQUE_KEY_COST.cost() - DICT_SQUASH_REPEATED_ACCESS_COST.cost()) as u64;
        (DICT_SQUASH_UNIQUE_KEY_COST.cost() - DICT_SQUASH_REPEATED_ACCESS_COST.cost()) as u64;
//}
}
//

///// Based on `cairo-lang-runner`'s implementation.
/// Based on `cairo-lang-runner`'s implementation.
/////
///
///// Source: <https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-runner/src/casm_run/mod.rs#L1946-L1948>
/// Source: <https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-runner/src/casm_run/mod.rs#L1946-L1948>
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__libfunc__debug__print(
pub unsafe extern "C" fn cairo_native__libfunc__debug__print(
//    target_fd: i32,
    target_fd: i32,
//    data: *const [u8; 32],
    data: *const [u8; 32],
//    len: u32,
    len: u32,
//) -> i32 {
) -> i32 {
//    let mut target = File::from_raw_fd(target_fd);
    let mut target = File::from_raw_fd(target_fd);
//

//    for i in 0..len as usize {
    for i in 0..len as usize {
//        let data = *data.add(i);
        let data = *data.add(i);
//

//        let value = Felt::from_bytes_le(&data);
        let value = Felt::from_bytes_le(&data);
//        if write!(target, "[DEBUG]\t{value:x}",).is_err() {
        if write!(target, "[DEBUG]\t{value:x}",).is_err() {
//            return 1;
            return 1;
//        };
        };
//

//        if data[..32]
        if data[..32]
//            .iter()
            .iter()
//            .copied()
            .copied()
//            .all(|ch| ch == 0 || ch.is_ascii_graphic() || ch.is_ascii_whitespace())
            .all(|ch| ch == 0 || ch.is_ascii_graphic() || ch.is_ascii_whitespace())
//        {
        {
//            let mut buf = [0; 31];
            let mut buf = [0; 31];
//            let mut len = 31;
            let mut len = 31;
//            for &ch in data.iter().take(31) {
            for &ch in data.iter().take(31) {
//                if ch != 0 {
                if ch != 0 {
//                    len -= 1;
                    len -= 1;
//                    buf[len] = ch;
                    buf[len] = ch;
//                }
                }
//            }
            }
//

//            if write!(
            if write!(
//                target,
                target,
//                " ('{}')",
                " ('{}')",
//                std::str::from_utf8_unchecked(&buf[len..])
                std::str::from_utf8_unchecked(&buf[len..])
//            )
            )
//            .is_err()
            .is_err()
//            {
            {
//                return 1;
                return 1;
//            }
            }
//        }
        }
//

//        if writeln!(target).is_err() {
        if writeln!(target).is_err() {
//            return 1;
            return 1;
//        };
        };
//    }
    }
//

//    // Avoid closing `stdout`.
    // Avoid closing `stdout`.
//    std::mem::forget(target);
    std::mem::forget(target);
//

//    0
    0
//}
}
//

///// Compute `pedersen(lhs, rhs)` and store it into `dst`.
/// Compute `pedersen(lhs, rhs)` and store it into `dst`.
/////
///
///// All its operands need the values in big endian.
/// All its operands need the values in big endian.
/////
///
///// # Panics
/// # Panics
/////
///
///// This function will panic if either operand is out of range for a felt.
/// This function will panic if either operand is out of range for a felt.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__libfunc__pedersen(
pub unsafe extern "C" fn cairo_native__libfunc__pedersen(
//    dst: *mut u8,
    dst: *mut u8,
//    lhs: *const u8,
    lhs: *const u8,
//    rhs: *const u8,
    rhs: *const u8,
//) {
) {
//    // Extract arrays from the pointers.
    // Extract arrays from the pointers.
//    let dst = slice::from_raw_parts_mut(dst, 32);
    let dst = slice::from_raw_parts_mut(dst, 32);
//    let lhs = slice::from_raw_parts(lhs, 32);
    let lhs = slice::from_raw_parts(lhs, 32);
//    let rhs = slice::from_raw_parts(rhs, 32);
    let rhs = slice::from_raw_parts(rhs, 32);
//

//    // Convert to FieldElement.
    // Convert to FieldElement.
//    let lhs = FieldElement::from_byte_slice_be(lhs).unwrap();
    let lhs = FieldElement::from_byte_slice_be(lhs).unwrap();
//    let rhs = FieldElement::from_byte_slice_be(rhs).unwrap();
    let rhs = FieldElement::from_byte_slice_be(rhs).unwrap();
//

//    // Compute pedersen hash and copy the result into `dst`.
    // Compute pedersen hash and copy the result into `dst`.
//    let res = starknet_crypto::pedersen_hash(&lhs, &rhs);
    let res = starknet_crypto::pedersen_hash(&lhs, &rhs);
//    dst.copy_from_slice(&res.to_bytes_be());
    dst.copy_from_slice(&res.to_bytes_be());
//}
}
//

///// Compute `hades_permutation(op0, op1, op2)` and replace the operands with the results.
/// Compute `hades_permutation(op0, op1, op2)` and replace the operands with the results.
/////
///
///// All operands need the values in big endian.
/// All operands need the values in big endian.
/////
///
///// # Panics
/// # Panics
/////
///
///// This function will panic if either operand is out of range for a felt.
/// This function will panic if either operand is out of range for a felt.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__libfunc__hades_permutation(
pub unsafe extern "C" fn cairo_native__libfunc__hades_permutation(
//    op0: *mut u8,
    op0: *mut u8,
//    op1: *mut u8,
    op1: *mut u8,
//    op2: *mut u8,
    op2: *mut u8,
//) {
) {
//    // Extract arrays from the pointers.
    // Extract arrays from the pointers.
//    let op0 = slice::from_raw_parts_mut(op0, 32);
    let op0 = slice::from_raw_parts_mut(op0, 32);
//    let op1 = slice::from_raw_parts_mut(op1, 32);
    let op1 = slice::from_raw_parts_mut(op1, 32);
//    let op2 = slice::from_raw_parts_mut(op2, 32);
    let op2 = slice::from_raw_parts_mut(op2, 32);
//

//    // Convert to FieldElement.
    // Convert to FieldElement.
//    let mut state = [
    let mut state = [
//        FieldElement::from_byte_slice_be(op0).unwrap(),
        FieldElement::from_byte_slice_be(op0).unwrap(),
//        FieldElement::from_byte_slice_be(op1).unwrap(),
        FieldElement::from_byte_slice_be(op1).unwrap(),
//        FieldElement::from_byte_slice_be(op2).unwrap(),
        FieldElement::from_byte_slice_be(op2).unwrap(),
//    ];
    ];
//

//    // Compute Poseidon permutation.
    // Compute Poseidon permutation.
//    starknet_crypto::poseidon_permute_comp(&mut state);
    starknet_crypto::poseidon_permute_comp(&mut state);
//

//    // Write back the results.
    // Write back the results.
//    op0.copy_from_slice(&state[0].to_bytes_be());
    op0.copy_from_slice(&state[0].to_bytes_be());
//    op1.copy_from_slice(&state[1].to_bytes_be());
    op1.copy_from_slice(&state[1].to_bytes_be());
//    op2.copy_from_slice(&state[2].to_bytes_be());
    op2.copy_from_slice(&state[2].to_bytes_be());
//}
}
//

///// Allocates a new dictionary. Internally a rust hashmap: `HashMap<[u8; 32], NonNull<()>`
/// Allocates a new dictionary. Internally a rust hashmap: `HashMap<[u8; 32], NonNull<()>`
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__alloc_dict() -> *mut std::ffi::c_void {
pub unsafe extern "C" fn cairo_native__alloc_dict() -> *mut std::ffi::c_void {
//    Box::into_raw(Box::<(HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64)>::default()) as _
    Box::into_raw(Box::<(HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64)>::default()) as _
//}
}
//

///// Frees the dictionary.
/// Frees the dictionary.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__dict_free(
pub unsafe extern "C" fn cairo_native__dict_free(
//    ptr: *mut (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64),
    ptr: *mut (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64),
//) {
) {
//    let mut map = Box::from_raw(ptr);
    let mut map = Box::from_raw(ptr);
//

//    // Free the entries manually.
    // Free the entries manually.
//    for (_, entry) in map.as_mut().0.drain() {
    for (_, entry) in map.as_mut().0.drain() {
//        libc::free(entry.as_ptr().cast());
        libc::free(entry.as_ptr().cast());
//    }
    }
//}
}
//

///// Gets the value for a given key, the returned pointer is null if not found.
/// Gets the value for a given key, the returned pointer is null if not found.
///// Increments the access count.
/// Increments the access count.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__dict_get(
pub unsafe extern "C" fn cairo_native__dict_get(
//    ptr: *mut (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64),
    ptr: *mut (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64),
//    key: &[u8; 32],
    key: &[u8; 32],
//) -> *mut std::ffi::c_void {
) -> *mut std::ffi::c_void {
//    let dict: &mut (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64) = &mut *ptr;
    let dict: &mut (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64) = &mut *ptr;
//    let map = &dict.0;
    let map = &dict.0;
//    dict.1 += 1;
    dict.1 += 1;
//

//    if let Some(v) = map.get(key) {
    if let Some(v) = map.get(key) {
//        v.as_ptr()
        v.as_ptr()
//    } else {
    } else {
//        std::ptr::null_mut()
        std::ptr::null_mut()
//    }
    }
//}
}
//

///// Inserts the provided key value. Returning the old one or nullptr if there was none.
/// Inserts the provided key value. Returning the old one or nullptr if there was none.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__dict_insert(
pub unsafe extern "C" fn cairo_native__dict_insert(
//    ptr: *mut (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64),
    ptr: *mut (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64),
//    key: &[u8; 32],
    key: &[u8; 32],
//    value: NonNull<std::ffi::c_void>,
    value: NonNull<std::ffi::c_void>,
//) -> *mut std::ffi::c_void {
) -> *mut std::ffi::c_void {
//    let dict = &mut *ptr;
    let dict = &mut *ptr;
//    let old_ptr = dict.0.insert(*key, value);
    let old_ptr = dict.0.insert(*key, value);
//

//    if let Some(v) = old_ptr {
    if let Some(v) = old_ptr {
//        v.as_ptr()
        v.as_ptr()
//    } else {
    } else {
//        std::ptr::null_mut()
        std::ptr::null_mut()
//    }
    }
//}
}
//

///// Compute the total gas refund for the dictionary at squash time.
/// Compute the total gas refund for the dictionary at squash time.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__dict_gas_refund(
pub unsafe extern "C" fn cairo_native__dict_gas_refund(
//    ptr: *const (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64),
    ptr: *const (HashMap<[u8; 32], NonNull<std::ffi::c_void>>, u64),
//) -> u64 {
) -> u64 {
//    let dict = &*ptr;
    let dict = &*ptr;
//    (dict.1 - dict.0.len() as u64) * *DICT_GAS_REFUND_PER_ACCESS
    (dict.1 - dict.0.len() as u64) * *DICT_GAS_REFUND_PER_ACCESS
//}
}
//

///// Compute `ec_point_from_x_nz(x)` and store it.
/// Compute `ec_point_from_x_nz(x)` and store it.
/////
///
///// # Panics
/// # Panics
/////
///
///// This function will panic if either operand is out of range for a felt.
/// This function will panic if either operand is out of range for a felt.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_from_x_nz(
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_from_x_nz(
//    mut point_ptr: NonNull<[[u8; 32]; 2]>,
    mut point_ptr: NonNull<[[u8; 32]; 2]>,
//) -> bool {
) -> bool {
//    let x = FieldElement::from_bytes_be(&{
    let x = FieldElement::from_bytes_be(&{
//        let mut data = point_ptr.as_ref()[0];
        let mut data = point_ptr.as_ref()[0];
//        data.reverse();
        data.reverse();
//        data
        data
//    })
    })
//    .unwrap();
    .unwrap();
//

//    match AffinePoint::from_x(x) {
    match AffinePoint::from_x(x) {
//        Some(mut point) => {
        Some(mut point) => {
//            // If y > PRIME/ 2 use PRIME - y
            // If y > PRIME/ 2 use PRIME - y
//            if point.y >= *HALF_PRIME {
            if point.y >= *HALF_PRIME {
//                point.y = -point.y
                point.y = -point.y
//            }
            }
//            point_ptr.as_mut()[1].copy_from_slice(&point.y.to_bytes_be());
            point_ptr.as_mut()[1].copy_from_slice(&point.y.to_bytes_be());
//            point_ptr.as_mut()[1].reverse();
            point_ptr.as_mut()[1].reverse();
//

//            true
            true
//        }
        }
//        None => false,
        None => false,
//    }
    }
//}
}
//

///// Compute `ec_point_try_new_nz(x)`.
/// Compute `ec_point_try_new_nz(x)`.
/////
///
///// # Panics
/// # Panics
/////
///
///// This function will panic if either operand is out of range for a felt.
/// This function will panic if either operand is out of range for a felt.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_try_new_nz(
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_try_new_nz(
//    point_ptr: NonNull<[[u8; 32]; 2]>,
    point_ptr: NonNull<[[u8; 32]; 2]>,
//) -> bool {
) -> bool {
//    let x = FieldElement::from_bytes_be(&{
    let x = FieldElement::from_bytes_be(&{
//        let mut data = point_ptr.as_ref()[0];
        let mut data = point_ptr.as_ref()[0];
//        data.reverse();
        data.reverse();
//        data
        data
//    })
    })
//    .unwrap();
    .unwrap();
//    let y = FieldElement::from_bytes_be(&{
    let y = FieldElement::from_bytes_be(&{
//        let mut data = point_ptr.as_ref()[1];
        let mut data = point_ptr.as_ref()[1];
//        data.reverse();
        data.reverse();
//        data
        data
//    })
    })
//    .unwrap();
    .unwrap();
//

//    AffinePoint::from_x(x).is_some_and(|point| y == point.y || y == -point.y)
    AffinePoint::from_x(x).is_some_and(|point| y == point.y || y == -point.y)
//}
}
//

///// Compute `ec_state_add(state, point)` and store the state back.
/// Compute `ec_state_add(state, point)` and store the state back.
/////
///
///// # Panics
/// # Panics
/////
///
///// This function will panic if either operand is out of range for a felt.
/// This function will panic if either operand is out of range for a felt.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add(
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add(
//    mut state_ptr: NonNull<[[u8; 32]; 4]>,
    mut state_ptr: NonNull<[[u8; 32]; 4]>,
//    point_ptr: NonNull<[[u8; 32]; 2]>,
    point_ptr: NonNull<[[u8; 32]; 2]>,
//) {
) {
//    let mut state = AffinePoint {
    let mut state = AffinePoint {
//        x: FieldElement::from_bytes_be(&{
        x: FieldElement::from_bytes_be(&{
//            let mut data = state_ptr.as_ref()[0];
            let mut data = state_ptr.as_ref()[0];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        y: FieldElement::from_bytes_be(&{
        y: FieldElement::from_bytes_be(&{
//            let mut data = state_ptr.as_ref()[1];
            let mut data = state_ptr.as_ref()[1];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        infinity: false,
        infinity: false,
//    };
    };
//    let point = AffinePoint {
    let point = AffinePoint {
//        x: FieldElement::from_bytes_be(&{
        x: FieldElement::from_bytes_be(&{
//            let mut data = point_ptr.as_ref()[0];
            let mut data = point_ptr.as_ref()[0];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        y: FieldElement::from_bytes_be(&{
        y: FieldElement::from_bytes_be(&{
//            let mut data = point_ptr.as_ref()[1];
            let mut data = point_ptr.as_ref()[1];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        infinity: false,
        infinity: false,
//    };
    };
//

//    state += &point;
    state += &point;
//

//    state_ptr.as_mut()[0].copy_from_slice(&state.x.to_bytes_be());
    state_ptr.as_mut()[0].copy_from_slice(&state.x.to_bytes_be());
//    state_ptr.as_mut()[1].copy_from_slice(&state.y.to_bytes_be());
    state_ptr.as_mut()[1].copy_from_slice(&state.y.to_bytes_be());
//

//    state_ptr.as_mut()[0].reverse();
    state_ptr.as_mut()[0].reverse();
//    state_ptr.as_mut()[1].reverse();
    state_ptr.as_mut()[1].reverse();
//}
}
//

///// Compute `ec_state_add_mul(state, scalar, point)` and store the state back.
/// Compute `ec_state_add_mul(state, scalar, point)` and store the state back.
/////
///
///// # Panics
/// # Panics
/////
///
///// This function will panic if either operand is out of range for a felt.
/// This function will panic if either operand is out of range for a felt.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add_mul(
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add_mul(
//    mut state_ptr: NonNull<[[u8; 32]; 4]>,
    mut state_ptr: NonNull<[[u8; 32]; 4]>,
//    scalar_ptr: NonNull<[u8; 32]>,
    scalar_ptr: NonNull<[u8; 32]>,
//    point_ptr: NonNull<[[u8; 32]; 2]>,
    point_ptr: NonNull<[[u8; 32]; 2]>,
//) {
) {
//    let mut state = AffinePoint {
    let mut state = AffinePoint {
//        x: FieldElement::from_bytes_be(&{
        x: FieldElement::from_bytes_be(&{
//            let mut data = state_ptr.as_ref()[0];
            let mut data = state_ptr.as_ref()[0];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        y: FieldElement::from_bytes_be(&{
        y: FieldElement::from_bytes_be(&{
//            let mut data = state_ptr.as_ref()[1];
            let mut data = state_ptr.as_ref()[1];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        infinity: false,
        infinity: false,
//    };
    };
//    let scalar = FieldElement::from_bytes_be(&{
    let scalar = FieldElement::from_bytes_be(&{
//        let mut data = *scalar_ptr.as_ref();
        let mut data = *scalar_ptr.as_ref();
//        data.reverse();
        data.reverse();
//        data
        data
//    })
    })
//    .unwrap();
    .unwrap();
//    let point = AffinePoint {
    let point = AffinePoint {
//        x: FieldElement::from_bytes_be(&{
        x: FieldElement::from_bytes_be(&{
//            let mut data = point_ptr.as_ref()[0];
            let mut data = point_ptr.as_ref()[0];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        y: FieldElement::from_bytes_be(&{
        y: FieldElement::from_bytes_be(&{
//            let mut data = point_ptr.as_ref()[1];
            let mut data = point_ptr.as_ref()[1];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        infinity: false,
        infinity: false,
//    };
    };
//

//    state += &(&point * &scalar.to_bits_le());
    state += &(&point * &scalar.to_bits_le());
//

//    state_ptr.as_mut()[0].copy_from_slice(&state.x.to_bytes_be());
    state_ptr.as_mut()[0].copy_from_slice(&state.x.to_bytes_be());
//    state_ptr.as_mut()[1].copy_from_slice(&state.y.to_bytes_be());
    state_ptr.as_mut()[1].copy_from_slice(&state.y.to_bytes_be());
//

//    state_ptr.as_mut()[0].reverse();
    state_ptr.as_mut()[0].reverse();
//    state_ptr.as_mut()[1].reverse();
    state_ptr.as_mut()[1].reverse();
//}
}
//

///// Compute `ec_state_try_finalize_nz(state)` and store the result.
/// Compute `ec_state_try_finalize_nz(state)` and store the result.
/////
///
///// # Panics
/// # Panics
/////
///
///// This function will panic if either operand is out of range for a felt.
/// This function will panic if either operand is out of range for a felt.
/////
///
///// # Safety
/// # Safety
/////
///
///// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
///// definitely unsafe to use manually.
/// definitely unsafe to use manually.
//#[no_mangle]
#[no_mangle]
//pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_try_finalize_nz(
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_try_finalize_nz(
//    mut point_ptr: NonNull<[[u8; 32]; 2]>,
    mut point_ptr: NonNull<[[u8; 32]; 2]>,
//    state_ptr: NonNull<[[u8; 32]; 4]>,
    state_ptr: NonNull<[[u8; 32]; 4]>,
//) -> bool {
) -> bool {
//    let state = AffinePoint {
    let state = AffinePoint {
//        x: FieldElement::from_bytes_be(&{
        x: FieldElement::from_bytes_be(&{
//            let mut data = state_ptr.as_ref()[0];
            let mut data = state_ptr.as_ref()[0];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        y: FieldElement::from_bytes_be(&{
        y: FieldElement::from_bytes_be(&{
//            let mut data = state_ptr.as_ref()[1];
            let mut data = state_ptr.as_ref()[1];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        infinity: false,
        infinity: false,
//    };
    };
//    let random = AffinePoint {
    let random = AffinePoint {
//        x: FieldElement::from_bytes_be(&{
        x: FieldElement::from_bytes_be(&{
//            let mut data = state_ptr.as_ref()[2];
            let mut data = state_ptr.as_ref()[2];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        y: FieldElement::from_bytes_be(&{
        y: FieldElement::from_bytes_be(&{
//            let mut data = state_ptr.as_ref()[3];
            let mut data = state_ptr.as_ref()[3];
//            data.reverse();
            data.reverse();
//            data
            data
//        })
        })
//        .unwrap(),
        .unwrap(),
//        infinity: false,
        infinity: false,
//    };
    };
//

//    if state.x == random.x && state.y == random.y {
    if state.x == random.x && state.y == random.y {
//        false
        false
//    } else {
    } else {
//        let point = &state - &random;
        let point = &state - &random;
//

//        point_ptr.as_mut()[0].copy_from_slice(&point.x.to_bytes_be());
        point_ptr.as_mut()[0].copy_from_slice(&point.x.to_bytes_be());
//        point_ptr.as_mut()[1].copy_from_slice(&point.y.to_bytes_be());
        point_ptr.as_mut()[1].copy_from_slice(&point.y.to_bytes_be());
//

//        point_ptr.as_mut()[0].reverse();
        point_ptr.as_mut()[0].reverse();
//        point_ptr.as_mut()[1].reverse();
        point_ptr.as_mut()[1].reverse();
//

//        true
        true
//    }
    }
//}
}
