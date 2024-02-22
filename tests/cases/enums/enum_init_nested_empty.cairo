enum OuterEnum {
    A: felt252,
    B: InnerEmptyEnum,
}

enum InnerEmptyEnum {
    A: ()
}

fn main() -> (OuterEnum, OuterEnum) {
    (
        OuterEnum::A(1234),
        OuterEnum::B(InnerEmptyEnum::A(())),
    )
}
