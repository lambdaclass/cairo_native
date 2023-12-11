use core::debug::PrintTrait;

enum MyEnum {
    V64: u64,
    V32: u32,
    V16: u16,
    V8: u8,
}

fn main(x: MyEnum) {
    match x {
        MyEnum::V64(x) => x.print(),
        MyEnum::V32(x) => x.print(),
        MyEnum::V16(x) => x.print(),
        MyEnum::V8(x) => x.print(),
    }
}
