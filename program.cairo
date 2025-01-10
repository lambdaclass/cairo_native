 use core::{dict::Felt252DictTrait, nullable::Nullable};
 #[derive(Drop)]
 enum MyEnum {
     A: u32,
     B: u64,
     C: u128,
 }
 fn run_test() -> Felt252Dict<Nullable<MyEnum>> {
     let mut x: Felt252Dict<Nullable<MyEnum>> = Default::default();
     x.insert(0, NullableTrait::new(MyEnum::A(1)));
     x.insert(1, NullableTrait::new(MyEnum::B(2)));
     x.insert(2, NullableTrait::new(MyEnum::C(3)));
     x
 }
