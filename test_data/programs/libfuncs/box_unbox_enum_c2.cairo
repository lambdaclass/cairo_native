use core::box::BoxTrait;

enum MyEnum {
    A: (),
    B: (),
}

fn run_test() -> MyEnum {
    let x = BoxTrait::new(MyEnum::B);
    x.unbox()
}
