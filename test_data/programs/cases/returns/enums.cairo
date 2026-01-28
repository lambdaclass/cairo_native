enum MyEnum {
    A: u8,
    B: u16,
}

fn main() -> MyEnum {
    MyEnum::B(5_u16)
}
