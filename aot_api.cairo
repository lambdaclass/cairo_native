use core::traits::Into;
use core::{array::ArrayTrait, debug::PrintTrait};

#[derive(Drop)]
struct MyStruct {
    a: felt252, // Felts within structs.
    c: u8, // Force some padding by introducing an `u8`
    b: u64, // just before an `u64`.
}

enum MyEnum {
    A: u64,
    B: u8,
    C: felt252,
}

fn invoke0() {
    'Hello, world!'.print();
}

fn invoke1_felt252(arg0: felt252) {
    arg0.print();
}

fn invoke1_u8(arg0: u8) {
    arg0.print();
}

fn invoke1_u16(arg0: u16) {
    arg0.print();
}

fn invoke1_u32(arg0: u32) {
    arg0.print();
}

fn invoke1_u64(arg0: u64) {
    arg0.print();
}

fn invoke1_u128(arg0: u128) {
    arg0.print();
}

fn invoke1_MyStruct(arg0: MyStruct) {
    // TODO: Find out why the print comes out as `(a, c, b)` instead of `(a, b, c)`.
    let mut data = ArrayTrait::new();
    data.append(arg0.a);
    data.append(arg0.b.into());
    data.append(arg0.c.into());
    data.print();
}

fn invoke1_Array_felt252(arg0: Array<felt252>) {
    arg0.print();
}

fn invoke1_MyEnum(arg0: MyEnum) {
    let (tag, value) = match arg0 {
        MyEnum::A(a) => (0, a.into()),
        MyEnum::B(b) => (1, b.into()),
        MyEnum::C(c) => (2, c),
    };

    let mut data = ArrayTrait::new();
    data.append(tag);
    data.append(value);
    data.print();
}

fn invoke2_MyEnum_MyStruct(arg0: MyEnum, arg1: MyStruct) {
    let (tag, value) = match arg0 {
        MyEnum::A(a) => (0, a.into()),
        MyEnum::B(b) => (1, b.into()),
        MyEnum::C(c) => (2, c),
    };

    let mut data = ArrayTrait::new();
    data.append(tag);
    data.append(value);
    data.append(arg1.a);
    data.append(arg1.b.into());
    data.append(arg1.c.into());
    data.print();
}

fn invoke5_u64_felt252_felt252_felt252_felt252(
    arg0: u64, arg1: felt252, arg2: felt252, arg3: felt252, arg4: felt252,
) {
    arg0.print();
    arg1.print();
    arg2.print();
    arg3.print();
    arg4.print();
}

fn invoke8_u64_u64_u64_u64_u64_u64_u64_u64(
    arg0: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64, arg6: u64, arg7: u64
) {
    arg0.print();
    arg1.print();
    arg2.print();
    arg3.print();
    arg4.print();
    arg5.print();
    arg6.print();
    arg7.print();
}

fn invoke9_u64_u64_u64_u64_u64_u64_u64_u64_u64(
    arg0: u64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    arg4: u64,
    arg5: u64,
    arg6: u64,
    arg7: u64,
    arg8: u64,
) {
    arg0.print();
    arg1.print();
    arg2.print();
    arg3.print();
    arg4.print();
    arg5.print();
    arg6.print();
    arg7.print();
    arg8.print();
}

fn invoke0_return1_felt252() -> felt252 {
    42
}

fn invoke0_return1_u64() -> u64 {
    42
}

fn invoke0_return1_tuple10_u64() -> (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) {
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
}
