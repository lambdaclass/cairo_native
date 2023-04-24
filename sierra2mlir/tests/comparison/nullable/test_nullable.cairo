use nullable::null;
use nullable::nullable_from_box;
use box::BoxTrait;

fn main() -> (Nullable<u8>, Nullable<u8>) {
    let a = null();
    let b: Box<u8> = BoxTrait::new(4_u8);
    let c = nullable_from_box(b);
    (a, c)
}
