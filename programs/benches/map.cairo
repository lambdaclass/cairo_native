// Logistic map implementation.
//
// The value would normally be a number between zero and one, however since floats are not
// available, zero corresponds to a felt zero and one to PRIME - 1.
fn iterate_map(r: felt252, x: felt252) -> felt252 {
    r * x * -x
}

fn main() -> felt252 {
    // Initial value.
    let mut x = 1234567890123456789012345678901234567890;

    // Iterate the map.
    let mut i = 10000; // 10000 causes stack overflow
    loop {
        x = iterate_map(4, x);

        if i == 0 {
            break x;
        }

        i = i - 1;
    }
}
