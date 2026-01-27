enum MyEnum {
    A: (),
    B: felt252,
}

fn main() -> MyEnum {
    MyEnum::B(1234)
}
