#[starknet::interface]
trait ISimpleStorage<TContractState> {
    fn get(self: @TContractState, x: felt252) -> felt252;
}

#[starknet::contract]
mod contract {
    #[storage]
    struct Storage {}

    #[abi(embed_v0)]
    impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
        fn get(self: @ContractState, x: felt252) -> felt252 {
            factorial(1, x)
        }
    }

    fn factorial(value: felt252, n: felt252) -> felt252 {
        if (n == 1) {
            value
        } else {
            factorial(value * n, n - 1)
        }
    }
}
