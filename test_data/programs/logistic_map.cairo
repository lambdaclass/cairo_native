fn iterate_map(r: felt252, x: felt252) -> felt252 {
    r * x * -x
}

// good default: 1000
fn run_test(mut i: felt252) -> felt252 {
    // Initial value.
    let mut x = 1234567890123456789012345678901234567890;

    // Iterate the map.
    loop {
        x = iterate_map(4, x);

        if i == 0 {
            break x;
        }

        i = i - 1;
    }
}
