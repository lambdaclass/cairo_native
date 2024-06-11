enum MyEnum {
    A: felt252,
    B: (felt252, felt252),
}

fn get_value(e: MyEnum) -> felt252 {
    match e {
        MyEnum::A(a) => a,
        MyEnum::B((x,y)) => x - y,
    }
}

fn main() -> (felt252, felt252) {
    (
        get_value(MyEnum::A(10)),
        get_value(MyEnum::B((20, 30))),
    )
}
