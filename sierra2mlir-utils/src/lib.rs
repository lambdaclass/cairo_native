use core::slice;
use starknet_crypto::FieldElement;

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
