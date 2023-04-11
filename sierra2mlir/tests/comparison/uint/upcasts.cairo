use integer::u8_to_felt252;
use integer::u16_to_felt252;
use integer::u32_to_felt252;
use integer::u64_to_felt252;
use integer::u128_to_felt252;

fn main() -> (
	felt252, felt252, felt252, felt252, felt252,
	felt252, felt252, felt252, felt252, felt252,
	felt252, felt252, felt252, felt252, felt252,
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
	)
}
