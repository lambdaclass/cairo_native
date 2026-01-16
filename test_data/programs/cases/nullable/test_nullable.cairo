use nullable::null;
use nullable::match_nullable;
use nullable::FromNullableResult;
use nullable::nullable_from_box;
use box::BoxTrait;

fn main() -> (Nullable<u8>, Option<u8>) {
    let a: Nullable<u8> = null();
    let b: Box<u8> = BoxTrait::new(4_u8);
    let c = nullable_from_box(b);
    let d = match match_nullable(c) {
      FromNullableResult::Null(_) => Option::Some(6_u8),
      FromNullableResult::NotNull(value) => Option::Some(value.unbox())
    };
    // can't return b to print because box types print their memory cell?
    (a, d)
}
