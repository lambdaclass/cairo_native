use core::starknet::class_hash::class_hash_const;
use debug::PrintTrait;

fn main() {
    let b = class_hash_const::<10>();
    let felt: felt252 = b.into();
    felt.print()
}
