enum MyEnum {
    A: u8,
    B: u16
}

fn my_enum() -> MyEnum {
    MyEnum::A(4_u8)
}

fn my_enum2() -> MyEnum {
    MyEnum::B(8_u16)
}
