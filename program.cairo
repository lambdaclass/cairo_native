use core::poseidon::PoseidonTrait;
use core::hash::{HashStateTrait, HashStateExTrait};
use core::testing::get_available_gas;

fn main() {
    let a: u128 = get_available_gas();
    let b = PoseidonTrait::new().update_with(a).finalize();
    assert(b != 0, '');
}
