use core::box::BoxTrait;

enum MyEnum {
    A: felt252,
    B: u128,
}

fn run_test() -> MyEnum {
    let x = BoxTrait::new(MyEnum::A(1234));
    x.unbox()
}