fn fib(n: felt252) -> felt252 {
    if n == 0 {
        1
    } else if n == 1 {
        1
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main() -> (felt252, felt252, felt252) {
    (
        fib(0),
        fib(1),
        // Careful increasing this, each increment roughly doubles runtime
        fib(10),
    )
}