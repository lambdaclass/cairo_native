use array::ArrayTrait;

#[derive(Drop)]
enum Math {
    Zero: (),
    One: (),
    Sum: Array::<felt252>,
}

fn foo(e: @Math) -> felt252 {
    match e {
        Math::Zero(_) => 0,
        Math::One(_) => 1,
        Math::Sum(numbers) => *numbers[0] + *numbers[1],
    }
}

fn main() -> (felt252, felt252) {
    let mut numbers = ArrayTrait::new();
    numbers.append(10);
    numbers.append(7);
    (
        foo(@Math::Zero(())),
        foo(@Math::Sum(numbers)),
    )
}
