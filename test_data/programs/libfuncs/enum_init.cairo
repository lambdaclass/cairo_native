enum MySmallEnum {
    A: felt252,
}

enum MyEnum {
    A: felt252,
    B: u8,
    C: u16,
    D: u32,
    E: u64,
}

fn run_test() -> (MySmallEnum, MyEnum, MyEnum, MyEnum, MyEnum, MyEnum) {
    (
        MySmallEnum::A(-1),
        MyEnum::A(5678),
        MyEnum::B(90),
        MyEnum::C(9012),
        MyEnum::D(34567890),
        MyEnum::E(1234567890123456),
    )
}
