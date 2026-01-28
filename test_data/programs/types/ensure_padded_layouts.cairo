#[derive(Drop)]
struct A {}
#[derive(Drop)]
struct B { a: u8 }
#[derive(Drop)]
struct C { a: u8, b: u16 }
#[derive(Drop)]
struct D { a: u16, b: u8 }

fn main(a: A, b: B, c: C, d: D) {}
