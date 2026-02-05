use nullable::null;
use nullable::match_nullable;
use nullable::FromNullableResult;
use nullable::nullable_from_box;
use box::BoxTrait;

fn run_test(x: u8) -> u8 {
    let b: Box<u8> = BoxTrait::new(x);
    let c = if x == 0 {
        null()
    } else {
        nullable_from_box(b)
    };
    let d = match match_nullable(c) {
        FromNullableResult::Null(_) => 99_u8,
        FromNullableResult::NotNull(value) => value.unbox()
    };
    d
}
