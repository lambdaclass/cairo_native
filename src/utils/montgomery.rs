use num_bigint::{BigInt, BigUint, ToBigInt};
use num_traits::{One, Zero};
use starknet_types_core::felt::Felt;

pub trait MontyBytes {
    fn to_bytes_le_raw(&self) -> [u8; 32];
}

impl MontyBytes for Felt {
    fn to_bytes_le_raw(&self) -> [u8; 32] {
        let limbs = self.to_raw();
        let mut buffer = [0; 32];

        for i in (0..4).rev() {
            let bytes = limbs[i].to_le_bytes();
            let init = (3 - i) * 8;
            buffer[init..init + 8].copy_from_slice(&bytes);
        }

        buffer
    }
}

/// Computes mudulus^{-1} mod 2^{64}.
///
/// This algorithm is mostly inspired from Lambaworks's u32 Montgomery
/// implementation:
/// https://github.com/lambdaclass/lambdaworks/blob/main/crates/math/src/field/fields/u32_montgomery_backend_prime_field.rs#L36
pub fn compute_mu_parameter(modulus: &BigUint) -> BigUint {
    let mut y = BigUint::one();
    let word_size = 64;
    let mut i: usize = 2;
    while i <= word_size {
        let mul_result = modulus * &y;
        if (mul_result << (word_size - i)) >> (word_size - i) != BigUint::one() {
            let shifted = BigUint::one() << (i - 1);

            y += shifted;
        }
        i += 1;
    }
    y
}

/// Computes 2^{2 * 384} mod modulus.
///
/// This algorithm is mostly inspired from Lambaworks's u32 Montgomery
/// implementation:
/// https://github.com/lambdaclass/lambdaworks/blob/main/crates/math/src/field/fields/u32_montgomery_backend_prime_field.rs#L57
pub fn compute_r2_parameter(modulus: &BigUint) -> BigUint {
    let word_size = 384;
    let mut l: usize = 0;

    while l < word_size && (modulus >> l) == BigUint::zero() {
        l += 1;
    }

    let mut c = BigUint::one() << l;

    let mut i: usize = 1;
    while i <= 2 * word_size - l {
        let double_c: BigUint = c << 1;

        c = if &double_c >= modulus {
            double_c - modulus
        } else {
            double_c
        };
        i += 1;
    }
    c
}

/// Computes the Montgomery reduction.
/// TODO: add docs.
/// Inspired in Lambdaworks's `montgomery_reduction`:
/// https://github.com/lambdaclass/lambdaworks/blob/main/crates/math/src/field/fields/u32_montgomery_backend_prime_field.rs#L285
pub fn monty_reduction(x: &BigUint, mu: &BigUint, modulus: &BigUint) -> BigUint {
    let x = &x.to_bigint().unwrap();
    let mu = &mu.to_bigint().unwrap();
    let modulus = &modulus.to_bigint().unwrap();

    // q = (x * mu) mod r.
    let q = (x * mu) % (BigInt::one() << 256);
    // m = q * modulus
    let m: BigInt = q * modulus;
    // y = (x - m) / r
    let y: BigInt = (x - m) >> 256i16;

    if y < BigInt::zero() {
        (y + modulus).to_biguint().unwrap()
    } else {
        y.to_biguint().unwrap()
    }
}

/// Computes the Montgomery transform operation.
/// TODO: add docs.
pub fn monty_transform(x: &BigUint, r2: &BigUint, mu: &BigUint, modulus: &BigUint) -> BigUint {
    let y = x * r2;
    monty_reduction(&y, mu, modulus)
}
