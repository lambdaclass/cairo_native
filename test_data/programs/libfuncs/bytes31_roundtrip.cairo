use core::bytes_31::{bytes31_try_from_felt252, bytes31_to_felt252};

fn run_test(value: felt252) -> felt252 {
    let a: bytes31 = bytes31_try_from_felt252(value).unwrap();
    bytes31_to_felt252(a)
}