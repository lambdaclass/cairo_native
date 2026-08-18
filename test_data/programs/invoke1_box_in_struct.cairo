#[derive(Drop)]
struct Wrapper {
    a: felt252,
    b: Box<felt252>,
}

fn main(x: Wrapper) -> felt252 {
    x.a + x.b.unbox()
}
