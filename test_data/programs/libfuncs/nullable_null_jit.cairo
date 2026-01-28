use nullable::null;
use nullable::match_nullable;
use nullable::FromNullableResult;
use nullable::nullable_from_box;
use box::BoxTrait;

fn run_test() -> Nullable<u8> {
    let a: Nullable<u8> = null();
    a
}
