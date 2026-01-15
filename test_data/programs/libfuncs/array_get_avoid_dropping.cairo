use core::{array::{array_append, array_at, array_new}, box::{into_box, unbox}};

fn run_test() -> @Box<felt252> {
    let mut x: Array<Box<felt252>> = array_new();
    array_append(ref x, into_box(42));

    unbox(array_at(@x, 0))
}