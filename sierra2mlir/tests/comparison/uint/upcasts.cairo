use integer::u8_to_felt252;
use integer::u16_to_felt252;
use integer::u32_to_felt252;
use integer::u64_to_felt252;
use integer::u128_to_felt252;
use integer::upcast;

fn main() -> (
    felt252, felt252, felt252, felt252, felt252,
    felt252, felt252, felt252, felt252, felt252,
    felt252, felt252, felt252, felt252, felt252,
    u8, u8, u8,
    u16, u16, u16,
    u32, u32, u32,
    u64, u64, u64,
    u128, u128, u128,
    u16, u16, u16,
    u32, u32, u32,
    u64, u64, u64,
    u128, u128, u128,
    u32, u32, u32,
    u64, u64, u64,
    u128, u128, u128,
    u64, u64, u64,
    u128, u128, u128,
    u128, u128, u128,
) {
    (
        u8_to_felt252(0_u8),
        u16_to_felt252(0_u16),
        u32_to_felt252(0_u32),
        u64_to_felt252(0_u64),
        u128_to_felt252(0_u128),
        u8_to_felt252(100_u8),
        u16_to_felt252(100_u16),
        u32_to_felt252(100_u32),
        u64_to_felt252(100_u64),
        u128_to_felt252(100_u128),
        u8_to_felt252(255_u8),
        u16_to_felt252(65535_u16),
        u32_to_felt252(4294967295_u32),
        u64_to_felt252(18446744073709551615_u64),
        u128_to_felt252(340282366920938463463374607431768211455_u128),
        // u8 to u8
        upcast(0_u8),
        upcast(100_u8),
        upcast(255_u8),
        // u8 to u16
        upcast(0_u8),
        upcast(100_u8),
        upcast(255_u8),
        // u8 to u32
        upcast(0_u8),
        upcast(100_u8),
        upcast(255_u8),
        // u8 to u64
        upcast(0_u8),
        upcast(100_u8),
        upcast(255_u8),
        // u8 to u128
        upcast(0_u8),
        upcast(100_u8),
        upcast(255_u8),
        // u16 to u16
        upcast(0_u16),
        upcast(100_u16),
        upcast(65535_u16),
        // u16 to u32
        upcast(0_u16),
        upcast(100_u16),
        upcast(65535_u16),
        // u16 to u64
        upcast(0_u16),
        upcast(100_u16),
        upcast(65535_u16),
        // u16 to u128
        upcast(0_u16),
        upcast(100_u16),
        upcast(65535_u16),
        // u32 to u32
        upcast(0_u32),
        upcast(100_u32),
        upcast(4294967295_u32),
        // u32 to u64
        upcast(0_u32),
        upcast(100_u32),
        upcast(4294967295_u32),
        // u32 to u128
        upcast(0_u32),
        upcast(100_u32),
        upcast(4294967295_u32),
        // u64 to u64
        upcast(0_u64),
        upcast(100_u64),
        upcast(18446744073709551615_u64),
        // to u128
        upcast(0_u64),
        upcast(100_u64),
        upcast(18446744073709551615_u64),
        // u128 to u128
        upcast(0_u128),
        upcast(100_u128),
        upcast(340282366920938463463374607431768211455_u128),
    )
}
