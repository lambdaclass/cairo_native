#[derive(Drop)]
struct MyStruct {
    a: u8,
    b: i16,
    c: felt252,
}

fn run_program() -> Felt252Dict<Nullable<MyStruct>> {
    let mut x: Felt252Dict<Nullable<MyStruct>> = Default::default();
    x.insert(0, Default::default());
    x.insert(1, NullableTrait::new(MyStruct { a: 0, b: 1, c: 2 }));
    x.insert(2, NullableTrait::new(MyStruct { a: 1, b: -2, c: 3 }));
    x.insert(3, NullableTrait::new(MyStruct { a: 2, b: 3, c: 4 }));
    x
}
