fn run_test() -> @Array<Array<felt252>> {
    let mut inputs: Array<Array<felt252>> = ArrayTrait::new();
    inputs.append(array![1, 2, 3]);
    inputs.append(array![4, 5, 6]);

    @inputs
}