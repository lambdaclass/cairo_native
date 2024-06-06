#[starknet::contract]
mod Factorial {
    #[storage]
    struct Storage {}

    #[external(v0)]
    fn factorial(self: @ContractState, n: felt252) -> felt252 {
        if (n == 0) {
            return 1;
        }
        n * factorial(self, n - 1)
    }
}
