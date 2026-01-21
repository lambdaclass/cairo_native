use array::ArrayTrait;

#[derive(Drop)]
enum Color {
    White: (),
    Black: (),
    Colorful: Array::<felt252>,
}

fn foo(e: @Color) -> felt252 {
    match e {
        Color::White(_) => 0,
        Color::Black(_) => 1,
        Color::Colorful(_) => 2,
    }
}

fn main() -> (felt252, felt252) {
    let mut numbers = ArrayTrait::new();
    (
        foo(@Color::White(())),
        foo(@Color::Colorful(numbers)),
    )
}
