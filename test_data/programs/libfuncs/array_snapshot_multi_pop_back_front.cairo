use array::ArrayTrait;

fn run_test() -> (Span<felt252>, @Box<[felt252; 2]>, @Box<[felt252; 2]>) {
    let mut numbers = array![1, 2, 3, 4, 5, 6].span();
    let popped_front = numbers.multi_pop_front::<2>().unwrap();
    let popped_back = numbers.multi_pop_back::<2>().unwrap();

    (numbers, popped_front, popped_back)
}
