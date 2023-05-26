fn iterate_map(r: felt252, x: felt252) -> felt252 {
    r * x * -x
}

fn loop_map(n: u64, r: felt252, x: felt252) -> felt252 {
    if n == 0 {
        x
    } else if n % 2 == 0 {
        let x = loop_map(n / 2, r, x);
        let x = loop_map(n / 2, r, x);
        x
    } else {
        loop_map(n - 1, r, iterate_map(r, x))
    }
}

fn main() -> felt252 {
    loop_map(10000, 4, 1234567890123456789012345678901234567890)
}
