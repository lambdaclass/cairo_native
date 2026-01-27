use array::ArrayTrait;

#[derive(Drop)]
struct A {
    a: Array::<felt252>,
    b: felt252,
}

fn bar(a: @Array::<felt252>, b: @felt252) -> felt252 {
    *a[0] + *b
}

fn main() -> felt252 {
    let mut numbers = ArrayTrait::new();
    numbers.append(6);
    let s = @A { a: numbers, b: 1}; 
    bar(s.a, s.b)
}
