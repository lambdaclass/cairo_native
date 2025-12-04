//! # Montgomery implementation for Felt252.
//!
//! This module holds utility functions for performing arithmetic operations
//! inside the Montgomery space.
//!
//! Representing felts in the Montgomery space allows for optimizations when
//! performing multiplication and division operations. This is because it
//! avoids having to perform modulo operations and even divisions. Montgomery
//! reduces these operations to shifts and simple arithmetic operation such as
//! additions and subtractions.
//!
//! The way this works is by representing a values as x' = x * r mod n. This
//! introduces a new constant `r` which, for performance reasons, it is defined
//! as r = 2^{k} where k should be big enough to satisfy r > n.
//!
//! For more information on check: https://en.wikipedia.org/wiki/Montgomery_modular_multiplication.

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

// R parameter for felts. R = 2^{256} which is the smallest power of 2 greater than prime.
pub static MONTY_R: LazyLock<BigUint> = LazyLock::new(|| BigUint::from(1u64) << 256);
// R2 parameter for felts. R2 = 2^{256 * 2} mod prime. This value is a U256 instead of a
// BigUint to integrate with lambdaworks with ease.
pub static MONTY_R2: LazyLock<U256> = LazyLock::new(|| {
    UnsignedInteger::from_hex_unchecked(
        "7FFD4AB5E008810FFFFFFFFFF6F800000000001330FFFFFFFFFFD737E000401",
    )
});
// MU parameter for felts. MU = -prime^{-1} mod 2^{64}. The variant is used to
// allow a better integration with lambdaworks.
// Check: https://github.com/lambdaclass/lambdaworks/blob/main/crates/math/src/field/fields/montgomery_backed_prime_fields.rs#L60
pub const MONTY_MU_U64: u64 = 18446744073709551615;
// MU parameter for felts. MU = prime^{-1} mod R.
pub static MONTY_MU_U256: LazyLock<BigUint> = LazyLock::new(|| {
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

/// Utility function to convert Felt bytes in Montgomery form into a Felt with
/// its correct representation.
pub fn felt_from_monty_bytes(value: &[u8; 32]) -> Felt {
    let value = U256::from_bytes_le(value).unwrap();
    Felt::from_raw(value.limbs)
}

/// Computes the Montgomery reduction (REDC).
///
/// Having a value `x' = x . r mod n`, the Montgomery reduction can be
/// interpreted as dividing `x by r mod n`, such that `REDC(x') = x`.
///
/// For more info on this operation check:
/// https://en.wikipedia.org/wiki/Montgomery_modular_multiplication#The_REDC_algorithm.
pub fn monty_reduction(x: &BigUint, modulus: &BigUint) -> Result<BigUint, CreationError> {
    let x = U256::from_hex(&x.to_str_radix(16))?;
    let modulus = U256::from_hex(&modulus.to_str_radix(16))?;

    let reduced = MontgomeryAlgorithms::cios(&x, &U256::from_u64(1), &modulus, &MONTY_MU_U64);

    Ok(BigUint::from_bytes_le(&reduced.to_bytes_le()))
}

/// Computes the Montgomery transform operation.
///
/// To efficiently perform this operation, a precomputed `r^{2}` value is used.
/// This way `x' = REDC(x * r^{2})`. Since we are multiplying by `r^{2}`, and we want
/// `x' = x * r mod n`, we need to apply a reduction after multiplication.
///
/// For more info on this operation check:
/// https://en.wikipedia.org/wiki/Montgomery_modular_multiplication#Arithmetic_in_Montgomery_form.
pub fn monty_transform(x: &BigUint, modulus: &BigUint) -> Result<BigUint, CreationError> {
    let x = U256::from_hex(&x.to_str_radix(16))?;
    let modulus = U256::from_hex(&modulus.to_str_radix(16))?;

    let reduced = MontgomeryAlgorithms::cios(&x, &MONTY_R2, &modulus, &MONTY_MU_U64);

    Ok(BigUint::from_bytes_le(&reduced.to_bytes_le()))
}

pub mod mlir {
    use crate::{
        error::Result,
        utils::{
            montgomery::{MONTY_MU_U256, MONTY_R, MONTY_R2},
            PRIME,
        },
    };
    use melior::{
        dialect::{arith, ods, scf},
        helpers::{ArithBlockExt, BuiltinBlockExt},
        ir::{r#type::IntegerType, Block, BlockLike, Location, Region, Type, Value, ValueLike},
        Context,
    };

    /// Computes Montgomery multiplication in MLIR.
    pub fn monty_mul<'c, 'a>(
        context: &'c Context,
        block: &'a Block<'c>,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        res_ty: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        let i512 = IntegerType::new(context, 512).into();

        let lhs = block.extui(lhs, i512, location)?;
        let rhs = block.extui(rhs, i512, location)?;

        let t = block.muli(lhs, rhs, location)?;

        Ok(monty_reduce(context, block, t, res_ty, location)?)
    }

    /// Computes Montgomery division in MLIR.
    pub fn monty_div<'c, 'a>(
        context: &'c Context,
        block: &'a Block<'c>,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        res_ty: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        let inv_rhs = monty_inverse(context, block, rhs, location)?;
        monty_mul(context, block, lhs, inv_rhs, res_ty, location)
    }

    /// Compute Montgomery modular inverse.
    ///
    /// The algorithm is given by B. S. Kaliski Jr. in "The Montgomery Inverse
    /// and Its Applications". The algorithm consists of two phases:
    ///     1. Compute x = a^{-1}2^{k} mod p, where n < k < 2n (denoted as
    ///        almost inverse).
    ///     2. Corrects the result from phase 1 so that x = a^{-1}2^{n} mod p.
    /// The algorithm can also be checked here:
    /// https://www.researchgate.net/publication/3044233_The_Montgomery_modular_inverse_-_Revisited.
    fn monty_inverse<'c, 'a>(
        context: &'c Context,
        block: &'a Block<'c>,
        value: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        let value = block.extui(value, IntegerType::new(context, 256).into(), location)?;
        let (r, k) = almost_inverse(context, block, value, location)?;
        let inverse = inverse_correction(context, block, r, k, location)?;

        // Since value is already in its Montgomery form, we currently that
        // inverse = MontyInv(value) = (value * r)^{-1} mod n. Since we need
        // inverse = value^{-1} * r mod n, we still need to perform one more
        // correction. So,
        // inverse = MontyProd(inverse, r^{2}) = value^{-1} * r mod n.
        let r2 = block.const_int_from_type(context, location, *MONTY_R2, inverse.r#type())?;
        monty_mul(context, block, inverse, r2, inverse.r#type(), location)
    }

    /// Performs the Montgomery inverse correction.
    ///
    /// This algorithm represented phase 2 of of B. S. Kaliski Jr.'s Montgomery
    /// inverse which returns (a * 2^{m})^{-1} mod n, where `a` is the value
    /// to invert and `m` the smallest value such that 2^{m} > n.
    fn inverse_correction<'c, 'a>(
        context: &'c Context,
        block: &'a Block<'c>,
        r: Value<'c, '_>,
        k: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        let i16 = IntegerType::new(context, 16).into();
        let i256 = IntegerType::new(context, 256).into();

        let k0 = block.const_int(context, location, 0, 256)?;
        let k0_i16 = block.const_int(context, location, 0, 16)?;
        let k1_i16 = block.const_int(context, location, 1, 16)?;
        let k1 = block.const_int(context, location, 1, 256)?;
        let k256 = block.const_int(context, location, 256, 16)?;

        let loop_limit = block.subi(k, k256, location)?;

        let result = block.append_operation(
            ods::scf::r#for(
                context,
                &[i256],
                k0_i16,
                loop_limit,
                k1_i16,
                &[r],
                {
                    let region = Region::new();
                    let loop_block =
                        region.append_block(Block::new(&[(i16, location), (i256, location)]));

                    let r = loop_block.arg(1)?;

                    let r_and_one = loop_block.andi(r, k1, location)?;
                    let is_r_even = loop_block.cmpi(
                        context,
                        arith::CmpiPredicate::Eq,
                        r_and_one,
                        k0,
                        location,
                    )?;

                    let next_r = loop_block.append_op_result(scf::r#if(
                        is_r_even,
                        &[i256],
                        {
                            let region = Region::new();
                            let block_then = region.append_block(Block::new(&[]));

                            let result = block_then.shrui(r, k1, location)?;

                            block_then.append_operation(scf::r#yield(&[result], location));

                            region
                        },
                        {
                            let region = Region::new();
                            let block_else = region.append_block(Block::new(&[]));

                            let prime =
                                block_else.const_int(context, location, PRIME.clone(), 256)?;

                            let result = block_else.addi(r, prime, location)?;
                            let result = block_else.shrui(result, k1, location)?;

                            block_else.append_operation(scf::r#yield(&[result], location));

                            region
                        },
                        location,
                    ))?;

                    loop_block.append_operation(scf::r#yield(&[next_r], location));

                    region
                },
                location,
            )
            .into(),
        );

        Ok(result.result(0)?.into())
    }

    /// Performs a first approach to the Montgomery Inverse.
    ///
    /// This algorithm represents phase 1 of B. S. Kaliski Jr.'s Montgomery
    /// inverse which returns `alm_inv = (a * 2^{k})^{-1} mod n`, where `a` is
    /// the value to invert and `k` a value such that `m < k < 2 * m`, being
    /// `m` the smallest value such that 2^{m} > n.
    fn almost_inverse<'c, 'a>(
        context: &'c Context,
        block: &'a Block<'c>,
        value: Value<'c, '_>,
        location: Location<'c>,
    ) -> Result<(Value<'c, 'a>, Value<'c, 'a>)> {
        let i16 = IntegerType::new(context, 16).into();
        let value_ty = value.r#type();

        let k0 = block.const_int_from_type(context, location, 0, value_ty)?;
        let k0_i16 = block.const_int(context, location, 0, 16)?;
        let prime = block.const_int_from_type(context, location, PRIME.clone(), value_ty)?;
        let k1 = block.const_int_from_type(context, location, 1, value_ty)?;
        let k1_i16 = block.const_int(context, location, 1, 16)?;

        let result = block.append_operation(scf::r#while(
            &[prime, value, k0, k1, k0_i16],
            &[value_ty, value_ty, value_ty, value_ty, i16],
            {
                let region = Region::new();
                let cond_block = region.append_block(Block::new(&[
                    (value_ty, location),
                    (value_ty, location),
                    (value_ty, location),
                    (value_ty, location),
                    (i16, location),
                ]));
                let u = cond_block.arg(0)?;
                let v = cond_block.arg(1)?;
                let r = cond_block.arg(2)?;
                let s = cond_block.arg(3)?;

                let u_is_even = {
                    let u_and_one = cond_block.andi(u, k1, location)?;
                    cond_block.cmpi(context, arith::CmpiPredicate::Eq, u_and_one, k0, location)?
                };

                // if u is even then
                //    u = u / 2
                //    s = 2 * s
                // else if v is even then
                //    v = v / 2
                //    s = 2 * s
                // else if u > v then
                //    u = (u − v) / 2
                //    r = r + s
                //    s = 2 * s
                // else if u <= v then
                //     v = (v − u) / 2
                //     s = r + s
                //     r = 2 * r
                let result = cond_block.append_operation(scf::r#if(
                    u_is_even,
                    &[value_ty, value_ty, value_ty, value_ty],
                    {
                        let region = Region::new();
                        let u_even_block = region.append_block(Block::new(&[]));

                        let u = u_even_block.shrui(u, k1, location)?;
                        let s = u_even_block.shli(s, k1, location)?;

                        u_even_block.append_operation(scf::r#yield(&[u, v, r, s], location));

                        region
                    },
                    {
                        let region = Region::new();
                        let u_not_even_block = region.append_block(Block::new(&[]));

                        let v_is_even = {
                            let v_and_one = u_not_even_block.andi(v, k1, location)?;
                            u_not_even_block.cmpi(
                                context,
                                arith::CmpiPredicate::Eq,
                                v_and_one,
                                k0,
                                location,
                            )?
                        };

                        let result = u_not_even_block.append_operation(scf::r#if(
                            v_is_even,
                            &[value_ty, value_ty, value_ty, value_ty],
                            {
                                let region = Region::new();
                                let v_even_block = region.append_block(Block::new(&[]));

                                let v = v_even_block.shrui(v, k1, location)?;
                                let r = v_even_block.shli(r, k1, location)?;

                                v_even_block
                                    .append_operation(scf::r#yield(&[u, v, r, s], location));

                                region
                            },
                            {
                                let region = Region::new();
                                let v_not_even_block = region.append_block(Block::new(&[]));

                                let is_u_gt_v = v_not_even_block.cmpi(
                                    context,
                                    arith::CmpiPredicate::Ugt,
                                    u,
                                    v,
                                    location,
                                )?;

                                let result = v_not_even_block.append_operation(scf::r#if(
                                    is_u_gt_v,
                                    &[value_ty, value_ty, value_ty, value_ty],
                                    {
                                        let region = Region::new();
                                        let u_gt_v_block = region.append_block(Block::new(&[]));

                                        let u = {
                                            let u_min_v = u_gt_v_block.subi(u, v, location)?;
                                            u_gt_v_block.shrui(u_min_v, k1, location)?
                                        };
                                        let r = u_gt_v_block.addi(r, s, location)?;
                                        let s = u_gt_v_block.shli(s, k1, location)?;

                                        u_gt_v_block.append_operation(scf::r#yield(
                                            &[u, v, r, s],
                                            location,
                                        ));

                                        region
                                    },
                                    {
                                        let region = Region::new();
                                        let v_ge_u_block = region.append_block(Block::new(&[]));

                                        let v = {
                                            let v_min_u = v_ge_u_block.subi(v, u, location)?;
                                            v_ge_u_block.shrui(v_min_u, k1, location)?
                                        };
                                        let s = v_ge_u_block.addi(r, s, location)?;
                                        let r = v_ge_u_block.shli(r, k1, location)?;

                                        v_ge_u_block.append_operation(scf::r#yield(
                                            &[u, v, r, s],
                                            location,
                                        ));

                                        region
                                    },
                                    location,
                                ));

                                let u = result.result(0)?.into();
                                let v = result.result(1)?.into();
                                let r = result.result(2)?.into();
                                let s = result.result(3)?.into();

                                v_not_even_block
                                    .append_operation(scf::r#yield(&[u, v, r, s], location));

                                region
                            },
                            location,
                        ));

                        let u = result.result(0)?.into();
                        let v = result.result(1)?.into();
                        let r = result.result(2)?.into();
                        let s = result.result(3)?.into();

                        u_not_even_block.append_operation(scf::r#yield(&[u, v, r, s], location));
                        region
                    },
                    location,
                ));

                let u = result.result(0)?.into();
                let v = result.result(1)?.into();
                let r = result.result(2)?.into();
                let s = result.result(3)?.into();
                let k = cond_block.addi(cond_block.arg(4)?, k1_i16, location)?;

                let is_v_gt_zero =
                    cond_block.cmpi(context, arith::CmpiPredicate::Ugt, v, k0, location)?;

                cond_block.append_operation(scf::condition(
                    is_v_gt_zero,
                    &[u, v, r, s, k],
                    location,
                ));

                region
            },
            {
                let region = Region::new();
                let loop_block = region.append_block(Block::new(&[
                    (value_ty, location),
                    (value_ty, location),
                    (value_ty, location),
                    (value_ty, location),
                    (i16, location),
                ]));

                let u = loop_block.arg(0)?;
                let v = loop_block.arg(1)?;
                let r = loop_block.arg(2)?;
                let s = loop_block.arg(3)?;
                let k = loop_block.arg(4)?;

                loop_block.append_operation(scf::r#yield(&[u, v, r, s, k], location));

                region
            },
            location,
        ));

        let (almost_inv, k) = {
            // if r >= p:
            //     r = r − p
            // else:
            //     r
            // return (p - r), k
            let k = result.result(4)?.into();
            let r = {
                let r = result.result(2)?.into();
                let r_wrapped = block.subi(r, prime, location)?;
                let r_ge_prime =
                    block.cmpi(context, arith::CmpiPredicate::Uge, r, prime, location)?;
                let r =
                    block.append_op_result(arith::select(r_ge_prime, r_wrapped, r, location))?;

                block.subi(prime, r, location)?
            };

            (r, k)
        };

        Ok((almost_inv, k))
    }

    /// Computes Montgomery reduction in MLIR.
    pub fn monty_reduce<'c, 'a>(
        context: &'c Context,
        block: &'a Block<'c>,
        x: Value<'c, '_>,
        res_ty: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        let x = block.extui(x, IntegerType::new(context, 512).into(), location)?;
        let mu = block.const_int(context, location, MONTY_MU_U256.clone(), 512)?;
        let r_minus_1 = block.const_int(context, location, MONTY_R.clone() - 1u8, 512)?;
        let k256 = block.const_int(context, location, 256, 512)?;
        let modulus = block.const_int(context, location, PRIME.clone(), 512)?;

        // q = (value * mu) mod r.
        let q = block.muli(x, mu, location)?;
        let q = block.andi(q, r_minus_1, location)?;
        // m = q * modulus.
        let m = block.muli(q, modulus, location)?;
        // y = (value - m) / r.
        let y = block.subi(x, m, location)?;
        let y = block.shrui(y, k256, location)?;
        // if (m > x):
        //     y = y + modulus
        let y_plus_mod = block.addi(y, modulus, location)?;

        let is_negative = block.cmpi(context, arith::CmpiPredicate::Ugt, m, x, location)?;

        let value = block.append_op_result(arith::select(is_negative, y_plus_mod, y, location))?;
        Ok(block.trunci(value, res_ty, location)?)
    }

    /// Computes to Montgomery space conversion in MLIR.
    pub fn monty_transform<'c, 'a>(
        context: &'c Context,
        block: &'a Block<'c>,
        x: Value<'c, '_>,
        res_ty: Type<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        let r2 = block.const_int(context, location, *MONTY_R2, 257)?;
        monty_mul(context, block, x, r2, res_ty, location)
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

        let felt = Felt::from(PRIME.clone());
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

        let felt = Felt::from(PRIME.clone()).to_biguint();
        let monty_felt = monty_transform(&felt, &PRIME).unwrap();
        let reduced_monty_felt = monty_reduction(&monty_felt, &PRIME).unwrap();

        assert_eq!(reduced_monty_felt, felt);
    }
}
