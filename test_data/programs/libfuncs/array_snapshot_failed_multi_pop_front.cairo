use array::ArrayTrait;

fn run_test() -> Span<felt252> {
    let mut numbers = array![1, 2].span();

    // should fail (return none)
    assert!(numbers.multi_pop_front::<3>().is_none());

    numbers
}