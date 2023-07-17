#![allow(non_snake_case)]

use cairo_felt::Felt252;
use cairo_lang_runner::short_string::as_cairo_short_string;
use starknet_crypto::FieldElement;
use starknet_curve::AffinePoint;
use std::{fs::File, io::Write, os::fd::FromRawFd, ptr::NonNull, slice};

/// Based on `cairo-lang-runner`'s implementation.
///
/// Source: https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-runner/src/casm_run/mod.rs#L1789-L1800
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__debug__print(
    target_fd: i32,
    data: *const [u8; 32],
    len: usize,
) -> i32 {
    let mut target = File::from_raw_fd(target_fd);

    for i in 0..len {
        let mut data = *data.add(i);
        data.reverse();

        let value = Felt252::from_bytes_be(&data);
        if let Some(shortstring) = as_cairo_short_string(&value) {
            if writeln!(
                target,
                "[DEBUG]\t{shortstring: <31}\t(raw: {})",
                value.to_bigint()
            )
            .is_err()
            {
                return 1;
            };
        } else if writeln!(target, "[DEBUG]\t{:<31}\t(raw: {})", ' ', value.to_bigint()).is_err() {
            return 1;
        }
    }
    if writeln!(target).is_err() {
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
    let lhs = FieldElement::from_byte_slice_be(lhs).unwrap();
    let rhs = FieldElement::from_byte_slice_be(rhs).unwrap();

    // Compute pedersen hash and copy the result into `dst`.
    let res = starknet_crypto::pedersen_hash(&lhs, &rhs);
    dst.copy_from_slice(&res.to_bytes_be());
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
    let x = FieldElement::from_bytes_be(&{
        let mut data = point_ptr.as_ref()[0];
        data.reverse();
        data
    })
    .unwrap();

    match AffinePoint::from_x(x) {
        Some(point) => {
            point_ptr.as_mut()[1].copy_from_slice(&point.y.to_bytes_be());
            point_ptr.as_mut()[1].reverse();

            true
        }
        None => false,
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
    point_ptr: NonNull<[[u8; 32]; 2]>,
) -> bool {
    let x = FieldElement::from_bytes_be(&{
        let mut data = point_ptr.as_ref()[0];
        data.reverse();
        data
    })
    .unwrap();
    let y = FieldElement::from_bytes_be(&{
        let mut data = point_ptr.as_ref()[1];
        data.reverse();
        data
    })
    .unwrap();

    AffinePoint::from_x(x).is_some_and(|point| y == point.y || y == -point.y)
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
    let mut state = AffinePoint {
        x: FieldElement::from_bytes_be(&{
            let mut data = state_ptr.as_ref()[0];
            data.reverse();
            data
        })
        .unwrap(),
        y: FieldElement::from_bytes_be(&{
            let mut data = state_ptr.as_ref()[1];
            data.reverse();
            data
        })
        .unwrap(),
        infinity: false,
    };
    let point = AffinePoint {
        x: FieldElement::from_bytes_be(&{
            let mut data = point_ptr.as_ref()[0];
            data.reverse();
            data
        })
        .unwrap(),
        y: FieldElement::from_bytes_be(&{
            let mut data = point_ptr.as_ref()[1];
            data.reverse();
            data
        })
        .unwrap(),
        infinity: false,
    };

    state += &point;

    state_ptr.as_mut()[0].copy_from_slice(&state.x.to_bytes_be());
    state_ptr.as_mut()[1].copy_from_slice(&state.y.to_bytes_be());

    state_ptr.as_mut()[0].reverse();
    state_ptr.as_mut()[1].reverse();
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
    let mut state = AffinePoint {
        x: FieldElement::from_bytes_be(&{
            let mut data = state_ptr.as_ref()[0];
            data.reverse();
            data
        })
        .unwrap(),
        y: FieldElement::from_bytes_be(&{
            let mut data = state_ptr.as_ref()[1];
            data.reverse();
            data
        })
        .unwrap(),
        infinity: false,
    };
    let scalar = FieldElement::from_bytes_be(&{
        let mut data = *scalar_ptr.as_ref();
        data.reverse();
        data
    })
    .unwrap();
    let point = AffinePoint {
        x: FieldElement::from_bytes_be(&{
            let mut data = point_ptr.as_ref()[0];
            data.reverse();
            data
        })
        .unwrap(),
        y: FieldElement::from_bytes_be(&{
            let mut data = point_ptr.as_ref()[1];
            data.reverse();
            data
        })
        .unwrap(),
        infinity: false,
    };

    state += &(&point * &scalar.to_bits_le());

    state_ptr.as_mut()[0].copy_from_slice(&state.x.to_bytes_be());
    state_ptr.as_mut()[1].copy_from_slice(&state.y.to_bytes_be());

    state_ptr.as_mut()[0].reverse();
    state_ptr.as_mut()[1].reverse();
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
    let state = AffinePoint {
        x: FieldElement::from_bytes_be(&{
            let mut data = state_ptr.as_ref()[0];
            data.reverse();
            data
        })
        .unwrap(),
        y: FieldElement::from_bytes_be(&{
            let mut data = state_ptr.as_ref()[1];
            data.reverse();
            data
        })
        .unwrap(),
        infinity: false,
    };
    let random = AffinePoint {
        x: FieldElement::from_bytes_be(&{
            let mut data = state_ptr.as_ref()[2];
            data.reverse();
            data
        })
        .unwrap(),
        y: FieldElement::from_bytes_be(&{
            let mut data = state_ptr.as_ref()[3];
            data.reverse();
            data
        })
        .unwrap(),
        infinity: false,
    };

    println!("state  = ({}, {})", state.x, state.y);
    println!("random = ({}, {})", random.x, random.y);

    if state.x == random.x && state.y == random.y {
        println!("Point is zero.");
        false
    } else {
        let point = &state - &random;

        println!("Point is ({}, {})", point.x, point.y);

        point_ptr.as_mut()[0].copy_from_slice(&point.x.to_bytes_be());
        point_ptr.as_mut()[1].copy_from_slice(&point.y.to_bytes_be());

        point_ptr.as_mut()[0].reverse();
        point_ptr.as_mut()[1].reverse();

        true
    }
}
