use integer::u8_to_felt252;
use integer::u16_to_felt252;
use integer::u32_to_felt252;
use integer::u64_to_felt252;
use integer::u128_to_felt252;

fn main() {
    let u8_value = 0_u8;
    let u16_value = 0_u16;
    let u32_value = 0_u32;
    let u64_value = 0_u64;
    let u128_value = 0_u128;

    let u8_felt: felt252 = u8_to_felt252(u8_value);
    let u16_felt: felt252 = u16_to_felt252(u16_value);
    let u32_felt: felt252 = u32_to_felt252(u32_value);
    let u64_felt: felt252 = u64_to_felt252(u64_value);
    let u128_felt: felt252 = u128_to_felt252(u128_value);
}
