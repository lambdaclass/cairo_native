use std::sync::LazyLock;

use lambdaworks_math::{
    errors::CreationError,
    traits::ByteConversion,
    unsigned_integer::{
        element::{UnsignedInteger, U256},
        montgomery::MontgomeryAlgorithms,
    },
};
use num_bigint::BigUint;
use starknet_types_core::felt::Felt;

pub static MONTY_R2: LazyLock<U256> = LazyLock::new(|| {
    UnsignedInteger::from_hex_unchecked(
        "7FFD4AB5E008810FFFFFFFFFF6F800000000001330FFFFFFFFFFD737E000401",
    )
});
pub static MONTY_MU: LazyLock<u64> = LazyLock::new(|| {
    "18446744073709551615"
        .parse()
        .expect("hardcoded Montgomery mu constant should be valid")
});

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

/// Computes the Montgomery reduction.
/// TODO: add docs.
/// Inspired in Lambdaworks's `montgomery_reduction`:
/// https://github.com/lambdaclass/lambdaworks/blob/main/crates/math/src/field/fields/u32_montgomery_backend_prime_field.rs#L285
pub fn monty_reduction(x: &BigUint, modulus: &BigUint) -> Result<BigUint, CreationError> {
    let x = U256::from_hex(&x.to_str_radix(16))?;
    let modulus = U256::from_hex(&modulus.to_str_radix(16))?;

    let reduced = MontgomeryAlgorithms::cios(&x, &U256::from_u64(1), &modulus, &MONTY_MU);

    Ok(BigUint::from_bytes_le(&reduced.to_bytes_le()))
}

/// Computes the Montgomery transform operation.
/// TODO: add docs.
pub fn monty_transform(x: &BigUint, modulus: &BigUint) -> Result<BigUint, CreationError> {
    let x = U256::from_hex(&x.to_str_radix(16))?;
    let modulus = U256::from_hex(&modulus.to_str_radix(16))?;

    let reduced = MontgomeryAlgorithms::cios(&x, &MONTY_R2, &modulus, &MONTY_MU);

    Ok(BigUint::from_bytes_le(&reduced.to_bytes_le()))
}

#[cfg(test)]
mod tests {
    use crate::utils::{
        montgomery::{monty_reduction, monty_transform, MontyBytes},
        PRIME,
    };
    use lambdaworks_math::{traits::ByteConversion, unsigned_integer::element::U256};
    use starknet_types_core::felt::Felt;

    #[test]
    fn felt_to_bytes_raw() {
        let felt = Felt::from(10);
        let bytes = felt.to_bytes_le_raw();
        let felt_from_raw = {
            let value = U256::from_bytes_le(&bytes).unwrap();
            Felt::from_raw(value.limbs)
        };

        assert_eq!(felt_from_raw, felt);

        let felt = Felt::from(-10);
        let bytes = felt.to_bytes_le_raw();
        let felt_from_raw = {
            let value = U256::from_bytes_le(&bytes).unwrap();
            Felt::from_raw(value.limbs)
        };

        assert_eq!(felt_from_raw, felt);

        let felt = Felt::from(&*PRIME);
        let bytes = felt.to_bytes_le_raw();
        let felt_from_raw = {
            let value = U256::from_bytes_le(&bytes).unwrap();
            Felt::from_raw(value.limbs)
        };

        assert_eq!(felt_from_raw, felt);
    }

    #[test]
    fn felt_to_monty_to_felt() {
        let felt = Felt::from(10).to_biguint();
        let monty_felt = monty_transform(&felt, &PRIME).unwrap();
        let reduced_monty_felt = monty_reduction(&monty_felt, &PRIME).unwrap();

        assert_eq!(reduced_monty_felt, felt);

        let felt = Felt::from(-10).to_biguint();
        let monty_felt = monty_transform(&felt, &PRIME).unwrap();
        let reduced_monty_felt = monty_reduction(&monty_felt, &PRIME).unwrap();

        assert_eq!(reduced_monty_felt, felt);

        let felt = Felt::from(&*PRIME).to_biguint();
        let monty_felt = monty_transform(&felt, &PRIME).unwrap();
        let reduced_monty_felt = monty_reduction(&monty_felt, &PRIME).unwrap();

        assert_eq!(reduced_monty_felt, felt);
    }
}
