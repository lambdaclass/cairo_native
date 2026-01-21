enum OuterEnum {
    A: felt252,
    B: InnerTupleEnum,
}

enum InnerTupleEnum {
    A: (u16, u8)
}

fn main() -> (OuterEnum, OuterEnum) {
    (
        OuterEnum::A(1234),
        OuterEnum::B(InnerTupleEnum::A((200, 190))),
    )
}
