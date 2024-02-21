enum SingleTupleEnum {
    A: (felt252, felt252)
}

fn main() -> SingleTupleEnum {
    SingleTupleEnum::A((1,2))
}
