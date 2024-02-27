enum OuterEnum {
    A: felt252,
    B: InnerCEnum,
}

enum InnerCEnum {
    A: (),
    B: (),
    C: (),
    D: (),
}

fn main() -> (OuterEnum, OuterEnum, OuterEnum, OuterEnum, OuterEnum) {
    (
        OuterEnum::A(1234),
        OuterEnum::B(InnerCEnum::A(())),
        OuterEnum::B(InnerCEnum::B(())),
        OuterEnum::B(InnerCEnum::C(())),
        OuterEnum::B(InnerCEnum::D(())),
    )
}
