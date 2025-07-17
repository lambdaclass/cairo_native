#[starknet::contract]
mod Fibonacci {
    #[storage]
    struct Storage { }

    #[external(v0)]
    fn fibonacci(ref self: ContractState, value: felt252) -> felt252 {
        return super::fibonacci(1, 1, value);
    }
}

fn fibonacci(a: felt252, b: felt252, n: felt252) -> felt252 {
    match n {
        0 => a,
        _ => fibonacci(b, a + b, n - 1),
    }
}
