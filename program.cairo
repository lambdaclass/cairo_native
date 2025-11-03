use core::gas::{GasReserve, gas_reserve_create, gas_reserve_utilize};

fn main() -> (u128, u128, u128) {
    let initial_gas = core::testing::get_available_gas();
    let reserve = OptionTrait::unwrap(gas_reserve_create(100));
    let final_gas = core::testing::get_available_gas();
    gas_reserve_utilize(reserve);

    (initial_gas, final_gas, initial_gas - final_gas)
}
