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
use num_traits::Num;
use starknet_types_core::felt::Felt;

pub const MONTY_R: LazyLock<BigUint> = LazyLock::new(|| BigUint::from(1u64) << 256);
pub static MONTY_R2: LazyLock<U256> = LazyLock::new(|| {
    UnsignedInteger::from_hex_unchecked(
        "7FFD4AB5E008810FFFFFFFFFF6F800000000001330FFFFFFFFFFD737E000401",
    )
});
pub const MONTY_MU_U64: u64 = 18446744073709551615;
pub const MONTY_MU_U256: LazyLock<BigUint> = LazyLock::new(|| {
    BigUint::from_str_radix(
        "f7ffffffffffffef000000000000000000000000000000000000000000000001",
        16,
    )
    .expect("hardcoded mu constant should be valid")
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
pub fn monty_reduction(x: &BigUint, modulus: &BigUint) -> Result<BigUint, CreationError> {
    let x = U256::from_hex(&x.to_str_radix(16))?;
    let modulus = U256::from_hex(&modulus.to_str_radix(16))?;

    let reduced = MontgomeryAlgorithms::cios(&x, &U256::from_u64(1), &modulus, &MONTY_MU_U64);

    Ok(BigUint::from_bytes_le(&reduced.to_bytes_le()))
}

/// Computes the Montgomery transform operation.
/// TODO: add docs.
pub fn monty_transform(x: &BigUint, modulus: &BigUint) -> Result<BigUint, CreationError> {
    let x = U256::from_hex(&x.to_str_radix(16))?;
    let modulus = U256::from_hex(&modulus.to_str_radix(16))?;

    let reduced = MontgomeryAlgorithms::cios(&x, &MONTY_R2, &modulus, &MONTY_MU_U64);

    Ok(BigUint::from_bytes_le(&reduced.to_bytes_le()))
}

pub mod mlir {
    use melior::{
        dialect::arith,
        helpers::{ArithBlockExt, BuiltinBlockExt},
        ir::{Block, Location, Value},
        Context,
    };
    use num_bigint::BigUint;

    use crate::{
        error::Result,
        utils::{
            montgomery::{MONTY_MU_U256, MONTY_R},
            PRIME,
        },
    };

    pub fn monty_mul<'c, 'a>(
        context: &'c Context,
        block: &'a Block<'c>,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        let unreduced_result = block.muli(lhs, rhs, location)?;
        monty_reduce(context, block, unreduced_result, &*PRIME, location)
    }

    pub fn monty_reduce<'c, 'a>(
        context: &'c Context,
        block: &'a Block<'c>,
        value: Value<'c, '_>,
        modulus: &BigUint,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        let mu = block.const_int(context, location, &*MONTY_MU_U256, 512)?;
        let modulus = block.const_int(context, location, modulus, 512)?;
        let r_minus_1 = block.const_int(context, location, &*MONTY_R - 1u8, 512)?;
        let k256 = block.const_int(context, location, 256, 512)?;

        let q = block.muli(value, mu, location)?;
        let q = block.andi(q, r_minus_1, location)?;
        let m = block.muli(q, modulus, location)?;
        let y = block.subi(value, m, location)?;
        let y = block.shrui(y, k256, location)?;
        let y_plus_mod = block.addi(y, modulus, location)?;

        let is_negative = block.cmpi(context, arith::CmpiPredicate::Ugt, m, y, location)?;

        Ok(block.append_op_result(arith::select(is_negative, y_plus_mod, y, location))?)
    }
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
