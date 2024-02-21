enum OuterEnum {
    A: felt252,
    B: InnerStructEnum,
}

enum InnerStructEnum {
    A: Payload
}

struct Payload {
    first: u16,
    second: u32,
}

fn main() -> (OuterEnum, OuterEnum) {
    (
        OuterEnum::A(1234),
        OuterEnum::B(InnerStructEnum::A(Payload{
            first: 10,
            second: 100,
        })),
    )
}
