use core::box::BoxTrait;

enum MyEnum {
    A: felt252,
}

fn run_test() -> MyEnum {
    let x = BoxTrait::new(MyEnum::A(1234));
    x.unbox()
}
