enum OuterEnum {
    A: felt252,
    B: InnerScalarEnum,
}

enum InnerScalarEnum {
    A: u16
}

fn main() -> (OuterEnum, OuterEnum) {
    (
        OuterEnum::A(1234),
        OuterEnum::B(InnerScalarEnum::A(100)),
    )
}
