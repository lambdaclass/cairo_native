use core::slice;
use starknet_crypto::FieldElement;
use starknet_curve::AffinePoint;

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
pub unsafe extern "C" fn sierra2mlir_util_pedersen(dst: *mut u8, lhs: *const u8, rhs: *const u8) {
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
pub unsafe extern "C" fn sierra2mlir_util_hades_permutation(
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
        FieldElement::from_byte_slice_be(op0).unwrap(),
        FieldElement::from_byte_slice_be(op1).unwrap(),
        FieldElement::from_byte_slice_be(op2).unwrap(),
    ];

    // Compute Poseidon permutation.
    starknet_crypto::poseidon_permute_comp(&mut state);

    // Write back the results.
    op0.copy_from_slice(&state[0].to_bytes_be());
    op1.copy_from_slice(&state[1].to_bytes_be());
    op2.copy_from_slice(&state[2].to_bytes_be());
}

/// Compute `ec_point_zero()`.
///
/// Its return values are stored in big endian.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn sierra2mlir_util_ec_point_zero(x: *mut u8, y: *mut u8, infinite: *mut u8) {
    // Extract arrays from the pointers.
    let x = slice::from_raw_parts_mut(x, 32);
    let y = slice::from_raw_parts_mut(y, 32);
    let infinite = slice::from_raw_parts_mut(infinite, 1);

    let ec_point =
        AffinePoint { x: FieldElement::default(), y: FieldElement::default(), infinity: true };

    // Compute pedersen hash and copy the result into `dst`.
    x.copy_from_slice(&ec_point.x.to_bytes_be());
    y.copy_from_slice(&ec_point.y.to_bytes_be());
    infinite.copy_from_slice(&(ec_point.infinity as u8).to_be_bytes());
}

/// Compute `ec_point_from_x_nz()`.
///
/// Its return values are stored in big endian.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn sierra2mlir_util_ec_point_from_x_nz(
    x: *mut [u8; 32],
    y: *mut [u8; 32],
    infinite: *mut u8,
) {
    let x_val = FieldElement::from_byte_slice_be(&*x).unwrap();

    let ec_point = AffinePoint::from_x(x_val);

    x.write(ec_point.x.to_bytes_be());
    y.write(ec_point.y.to_bytes_be());
    infinite.write(ec_point.infinity as u8);
}

/// Compute `ec_point_try_new_nz()`.
///
/// Its return values are stored in big endian.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn sierra2mlir_util_ec_point_try_new_nz(
    x: *const [u8; 32],
    y: *const [u8; 32],
) -> i32 {
    let x_val = FieldElement::from_byte_slice_be(&*x).unwrap();
    let y_val = FieldElement::from_byte_slice_be(&*y).unwrap();

    let ec_point = AffinePoint::from_x(x_val);

    (ec_point.infinity || (ec_point.y != y_val && ec_point.y != -y_val)) as i32
}

/// Compute `ec_state_add()`.
///
/// Its return values are stored in big endian.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn sierra2mlir_util_ec_state_add(
    state_x: *mut [u8; 32],
    state_y: *mut [u8; 32],
    point_x: *const [u8; 32],
    point_y: *const [u8; 32],
    point_infinity: *const i8,
) {
    let state = AffinePoint {
        x: FieldElement::from_bytes_be(&*state_x).unwrap(),
        y: FieldElement::from_bytes_be(&*state_y).unwrap(),
        infinity: false,
    };
    let point = AffinePoint {
        x: FieldElement::from_bytes_be(&*point_x).unwrap(),
        y: FieldElement::from_bytes_be(&*point_y).unwrap(),
        infinity: *point_infinity != 0,
    };

    let next_state = &state + &point;
    state_x.write(next_state.x.to_bytes_be());
    state_y.write(next_state.y.to_bytes_be());
}

/// Compute `ec_state_add_mul()`.
///
/// Its return values are stored in big endian.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn sierra2mlir_util_ec_state_add_mul(
    state_x: *mut [u8; 32],
    state_y: *mut [u8; 32],
    value: *const [u8; 32],
    point_x: *const [u8; 32],
    point_y: *const [u8; 32],
    point_infinity: *const i8,
) {
    let state = AffinePoint {
        x: FieldElement::from_bytes_be(&*state_x).unwrap(),
        y: FieldElement::from_bytes_be(&*state_y).unwrap(),
        infinity: false,
    };
    let point = AffinePoint {
        x: FieldElement::from_bytes_be(&*point_x).unwrap(),
        y: FieldElement::from_bytes_be(&*point_y).unwrap(),
        infinity: *point_infinity != 0,
    };

    let value = FieldElement::from_bytes_be(&*value).unwrap();

    let next_state = &state + &(&point * &value.to_bits_le());
    state_x.write(next_state.x.to_bytes_be());
    state_y.write(next_state.y.to_bytes_be());
}

/// Compute `ec_state_try_finalize_nz()`.
///
/// Its return values are stored in big endian.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn sierra2mlir_util_ec_state_try_finalize_nz(
    state_x: *mut [u8; 32],
    state_y: *mut [u8; 32],
    point_x: *const [u8; 32],
    point_y: *const [u8; 32],
) -> i32 {
    let state = AffinePoint {
        x: FieldElement::from_bytes_be(&*state_x).unwrap(),
        y: FieldElement::from_bytes_be(&*state_y).unwrap(),
        infinity: false,
    };
    let point = AffinePoint {
        x: FieldElement::from_bytes_be(&*point_x).unwrap(),
        y: FieldElement::from_bytes_be(&*point_y).unwrap(),
        infinity: false,
    };

    if state.x == point.x {
        assert_eq!(state.y, point.y);
        return 1;
    }

    let numerator = state.y + point.y;
    let denominator = state.x - point.x;

    let slope = numerator.floor_div(denominator);
    // let slope = numerator * denominator.invert().unwrap();
    let slope2 = slope * slope;

    let sum_x = state.x + point.x;
    let res_x = slope2 - sum_x;

    let x_diff = state.x - res_x;
    let slope_times_x_change = slope * x_diff;
    let res_y = slope_times_x_change - state.y;

    println!("{}", res_x);
    println!("{}", res_y);
    state_x.write(res_x.to_bytes_be());
    state_y.write(res_y.to_bytes_be());
    0
}
