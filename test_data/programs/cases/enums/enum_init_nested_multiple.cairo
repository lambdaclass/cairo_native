enum OuterEnum {
    A: felt252,
    B: InnerMultipleEnum,
}

enum InnerMultipleEnum {
    A: felt252,
    B: (u128, felt252),
}

fn main() -> (OuterEnum, OuterEnum, OuterEnum) {
    (
        OuterEnum::A(1234),
        OuterEnum::B(InnerMultipleEnum::A(283947)),
        OuterEnum::B(InnerMultipleEnum::B((1000, 10000))),
    )
}
