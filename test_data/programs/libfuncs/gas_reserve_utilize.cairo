use core::gas::{GasReserve, gas_reserve_create, gas_reserve_utilize};

fn run_test(amount: u128) -> u128 {
    let initial_gas = core::testing::get_available_gas();
    let reserve = gas_reserve_create(amount).unwrap();
    gas_reserve_utilize(reserve);
    let final_gas = core::testing::get_available_gas();

    initial_gas - final_gas
}