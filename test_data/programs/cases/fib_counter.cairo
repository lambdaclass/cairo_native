// Calculates fib...
fn fib(a: felt252, b: felt252, n: felt252) -> (felt252, felt252) {
    match n {
        0 => (1, 2),
        _ => {
            let (v, count) = fib(b, a + b, n - 1);
            (v, count + 1)
        },
    }
}

fn main() -> (felt252, felt252) {
    return fib(1, 1, 3);
}