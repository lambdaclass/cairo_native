mod common;
use crate::common::{felt, run_native_program};
use common::load_cairo;
use serde_json::json;

#[test]
fn fib() {
    let (source, program, runner) = load_cairo! {
        fn fib(a: felt252, b: felt252, n: felt252) -> felt252 {
            match n {
                0 => a,
                _ => fib(b, a + b, n - 1),
            }
        }

        fn run_test() -> felt252 {
            fib(0, 1, 10)
        }

    };

    let result = run_native_program(&(source, program), "run_test", json!([null, 0]));
    assert_eq!(result, json!([felt("55")]));
}
