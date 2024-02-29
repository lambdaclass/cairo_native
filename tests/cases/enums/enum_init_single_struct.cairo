enum SingleStructEnum {
    A: Payload
}

struct Payload {
    first: u8,
    second: felt252,
}

fn main() -> SingleStructEnum {
    SingleStructEnum::A(Payload {
        first: 12,
        second: 39845,
    })
}
