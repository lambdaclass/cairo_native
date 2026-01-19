fn run_program() -> Felt252Dict<u128> {
    let mut x: Felt252Dict<u128> = Default::default();
    x.insert(0, 0_u128);
    x.insert(1, 1_u128);
    x.insert(2, 2_u128);
    x.insert(3, 3_u128);
    x
}