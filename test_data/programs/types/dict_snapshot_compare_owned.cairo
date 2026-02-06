fn run_test() -> Felt252Dict<Nullable<Array<u32>>> {
    let mut dict: Felt252Dict<Nullable<Array<u32>>> = Default::default();
    dict.insert(2, NullableTrait::new(array![3, 4]));

    dict
}
