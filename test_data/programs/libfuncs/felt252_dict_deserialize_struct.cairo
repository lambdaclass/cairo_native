use core::{dict::Felt252DictTrait, nullable::Nullable};

fn run_test() -> Felt252Dict<Nullable<(u32, u64, u128)>> {
    let mut x: Felt252Dict<Nullable<(u32, u64, u128)>> = Default::default();
    x.insert(0, NullableTrait::new((1_u32, 2_u64, 3_u128)));
    x.insert(1, NullableTrait::new((2_u32, 3_u64, 4_u128)));
    x.insert(2, NullableTrait::new((3_u32, 4_u64, 5_u128)));
    x
}