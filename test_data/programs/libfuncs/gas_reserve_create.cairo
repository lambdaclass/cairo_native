use core::gas::{GasReserve, gas_reserve_create, gas_reserve_utilize};

fn run_test_1() -> Option<GasReserve> {
    gas_reserve_create(100)
}

fn run_test_2(amount: u128) -> u128 {
    let initial_gas = core::testing::get_available_gas();
    let reserve = gas_reserve_create(amount).unwrap();
    let final_gas = core::testing::get_available_gas();
    gas_reserve_utilize(reserve);

    initial_gas - final_gas
}
