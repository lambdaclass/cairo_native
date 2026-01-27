use core::num::traits::WideMul;

fn main() -> (
    u16, u16, u16,
    u16, u16, u16,
    u16, u16, u16,

    u32, u32, u32,
    u32, u32, u32,
    u32, u32, u32,

    u64, u64, u64,
    u64, u64, u64,
    u64, u64, u64,

    u128, u128, u128,
    u128, u128, u128,
    u128, u128, u128,

    u256, u256, u256,
    u256, u256, u256,
    u256, u256, u256,
) {
    (
        WideMul::wide_mul(0_u8, 0_u8),
        WideMul::wide_mul(0_u8, 10_u8),
        WideMul::wide_mul(0_u8, 255_u8),
        WideMul::wide_mul(10_u8, 0_u8),
        WideMul::wide_mul(10_u8, 10_u8),
        WideMul::wide_mul(10_u8, 255_u8),
        WideMul::wide_mul(255_u8, 0_u8),
        WideMul::wide_mul(255_u8, 10_u8),
        WideMul::wide_mul(255_u8, 255_u8),

        WideMul::wide_mul(0_u16, 0_u16),
        WideMul::wide_mul(0_u16, 1000_u16),
        WideMul::wide_mul(0_u16, 65535_u16),
        WideMul::wide_mul(1000_u16, 0_u16),
        WideMul::wide_mul(1000_u16, 1000_u16),
        WideMul::wide_mul(1000_u16, 65535_u16),
        WideMul::wide_mul(65535_u16, 0_u16),
        WideMul::wide_mul(65535_u16, 10_u16),
        WideMul::wide_mul(65535_u16, 65535_u16),

        WideMul::wide_mul(0_u32, 0_u32),
        WideMul::wide_mul(0_u32, 100000_u32),
        WideMul::wide_mul(0_u32, 4294967295_u32),
        WideMul::wide_mul(100000_u32, 0_u32),
        WideMul::wide_mul(100000_u32, 100000_u32),
        WideMul::wide_mul(100000_u32, 4294967295_u32),
        WideMul::wide_mul(4294967295_u32, 0_u32),
        WideMul::wide_mul(4294967295_u32, 10_u32),
        WideMul::wide_mul(4294967295_u32, 4294967295_u32),

        WideMul::wide_mul(0_u64, 0_u64),
        WideMul::wide_mul(0_u64, 10000000000_u64),
        WideMul::wide_mul(0_u64, 18446744073709551615_u64),
        WideMul::wide_mul(10000000000_u64, 0_u64),
        WideMul::wide_mul(10000000000_u64, 10000000000_u64),
        WideMul::wide_mul(10000000000_u64, 18446744073709551615_u64),
        WideMul::wide_mul(18446744073709551615_u64, 0_u64),
        WideMul::wide_mul(18446744073709551615_u64, 10_u64),
        WideMul::wide_mul(18446744073709551615_u64, 18446744073709551615_u64),

        WideMul::wide_mul(0_u128, 0_u128),
        WideMul::wide_mul(0_u128, 100000000000000000000_u128),
        WideMul::wide_mul(0_u128, 340282366920938463463374607431768211455_u128),
        WideMul::wide_mul(100000000000000000000_u128, 0_u128),
        WideMul::wide_mul(100000000000000000000_u128, 100000000000000000000_u128),
        WideMul::wide_mul(100000000000000000000_u128, 340282366920938463463374607431768211455_u128),
        WideMul::wide_mul(340282366920938463463374607431768211455_u128, 0_u128),
        WideMul::wide_mul(340282366920938463463374607431768211455_u128, 10_u128),
        WideMul::wide_mul(340282366920938463463374607431768211455_u128, 340282366920938463463374607431768211455_u128),
    )
}
