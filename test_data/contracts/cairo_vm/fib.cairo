#[starknet::contract]
mod Fibonacci {
    #[storage]
    struct Storage {}

    #[external(v0)]
    fn fib(self: @ContractState, a: felt252, b: felt252, n: felt252) -> felt252 {
        match n {
            0 => a,
            _ => fib(self, b, a + b, n - 1),
        }
    }
}
