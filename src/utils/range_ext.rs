use crate::utils::{get_integer_layout, PRIME};
use cairo_lang_sierra::extensions::utils::Range;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{Euclid, One};

pub trait RangeExt {
    /// Width in bits when the offset is zero (aka. the natural representation).
    fn zero_based_bit_width(&self) -> u32;
    /// Width in bits when the offset is not necessarily zero (aka. the compact representation).
    fn repr_bit_width(&self) -> u32;
    /// Encode a value into the compact representation: the stored bits are
    /// `(value - lower) mod PRIME`, laid out little-endian over the full
    /// `get_integer_layout(repr_bit_width())` size. Returns `None` when the
    /// value is not within the range.
    fn repr_encode(&self, value: &BigInt) -> Option<Vec<u8>>;
    /// Decode raw stored bits back into the value they represent: mask to
    /// `repr_bit_width()` bits and add `lower`.
    fn repr_decode(&self, raw: BigUint) -> BigInt;
}

impl RangeExt for Range {
    fn zero_based_bit_width(&self) -> u32 {
        // Formula for unsigned integers:
        //     x.bits()
        //
        // Formula for signed values (n-bit two's complement holds
        // [-2^(n-1), 2^(n-1) - 1]):
        //   - Positive: x.magnitude().bits() + 1
        //   - Negative: (x.magnitude() - BigUint::one()).bits() + 1
        //   - Zero: 0

        let width = if self.lower.sign() == Sign::Minus {
            let lower_width = (self.lower.magnitude() - BigUint::one()).bits() + 1;
            let upper_width = {
                let upper = &self.upper - &BigInt::one();
                match upper.sign() {
                    Sign::Minus => (upper.magnitude() - BigUint::one()).bits() + 1,
                    Sign::NoSign => 0,
                    Sign::Plus => upper.magnitude().bits() + 1,
                }
            };

            lower_width.max(upper_width) as u32
        } else {
            (&self.upper - &BigInt::one()).bits() as u32
        };

        // FIXME: Workaround for segfault in canonicalization (including LLVM 19).
        width.max(1)
    }

    fn repr_bit_width(&self) -> u32 {
        // FIXME: Workaround for segfault in canonicalization (including LLVM 19).
        ((self.size() - BigInt::one()).bits() as u32).max(1)
    }

    fn repr_encode(&self, value: &BigInt) -> Option<Vec<u8>> {
        // The subtraction is felt arithmetic so that ranges with a negative
        // lower bound round-trip: values are field elements in `[0, PRIME)`,
        // and `repr_decode`'s addition wraps the same way.
        let prime = BigInt::from_biguint(Sign::Plus, PRIME.clone());
        let stored = (value - &self.lower).rem_euclid(&prime);
        if stored >= self.size() {
            return None;
        }

        let mut bytes = stored.magnitude().to_bytes_le();
        bytes.resize(get_integer_layout(self.repr_bit_width()).size(), 0);
        Some(bytes)
    }

    fn repr_decode(&self, raw: BigUint) -> BigInt {
        let mask = (BigUint::one() << self.repr_bit_width()) - BigUint::one();
        BigInt::from_biguint(Sign::Plus, raw & mask) + &self.lower
    }
}
