use core::box::BoxTrait;

enum MyEnum {
    A: (),
    B: (),
}

fn run_test() -> MyEnum {
    let x = BoxTrait::new(MyEnum::A);
    x.unbox()
}
