use array::ArrayTrait;

fn run_test() -> (Span<felt252>, @Box<[felt252; 3]>) {
    let mut numbers = array![1, 2, 3, 4, 5, 6].span();
    let popped = numbers.multi_pop_front::<3>().unwrap();

    (numbers, popped)
}