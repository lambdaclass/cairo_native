use core::num::traits::WrappingSub;

fn main() -> (
    (u16, u16, u16),
    (u32, u32, u32),
    (u64, u64, u64),
    (u128, u128, u128),
) {
    (
        (
            6_u16 - 4_u16,
            0_u16.wrapping_sub(2_u16),
            50_u16 - 2_u16,
        ),
        (
            6_u32 - 4_u32,
            0_u32.wrapping_sub(2_u32),
            50_u32 - 2_u32,
        ),
        (
            6_u64 - 4_u64,
            0_u64.wrapping_sub(2_u64),
            50_u64 - 2_u64,
        ),
        (
            6_u128 - 4_u128,
            2_u128 - 1_u128, // no wrapping sub for u128
            50_u128 - 2_u128,
        ),
    )
}
