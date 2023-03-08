fn fib(a: felt, b: felt, n: felt) -> felt {
    match n {
        0 => a,
        _ => fib(b, a + b, n - 1),
    }
}
fn fib_mid(n: felt) {
    match n {
        0 => (),
        _ => {
            fib(0, 1, 500);
            fib_mid(n - 1);
        },
    }
}
fn main(a: felt) {
    fib_mid(100);
}
