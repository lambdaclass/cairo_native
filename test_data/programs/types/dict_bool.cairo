fn run_program() -> Felt252Dict<bool> {
    let mut x: Felt252Dict<bool> = Default::default();
    x.insert(0, false);
    x.insert(1, true);
    x
}
