struct MyStruct {
    a: felt252
}

enum MyEnum {
    VariantA: MyStruct,
    VariantB: u64
}

fn main(x: felt252) -> MyEnum {
    MyEnum::VariantA(MyStruct {
        a: x
    })
}
