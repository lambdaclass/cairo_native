use integer::u16_wrapping_add;
use integer::u32_wrapping_add;
use integer::u64_wrapping_add;
use integer::u128_wrapping_add;

fn main() -> (
    (u16, u16, u16),
    (u32, u32, u32),
    (u64, u64, u64),
    (u128, u128, u128),
) {
	(
        (
            4_u16 + 6_u16,
            u16_wrapping_add(60000_u16, 20000_u16),
            50_u16 + 2_u16,
        ),
        (
            4_u32 + 6_u32,
            u32_wrapping_add(4294967293_u32, 10_u32),
            50_u32 + 2_u32,
        ),
        (
            4_u64 + 6_u64,
            u64_wrapping_add(18446744073709551613_u64, 10_u64),
            50_u64 + 2_u64,
        ),
        (
            4_u128 + 6_u128,
            u128_wrapping_add(340282366920938463463374607431768211453_u128, 10_u128),
            50_u128 + 2_u128,
        ),
    )
}
