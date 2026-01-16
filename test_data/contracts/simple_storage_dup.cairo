#[starknet::interface]
trait ISimpleStorage<TContractState> {
    fn get(self: @TContractState, x: felt252) -> (felt252, felt252);
}

#[starknet::contract]
mod contract {
    #[storage]
    struct Storage {}

    #[abi(embed_v0)]
    impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
        fn get(self: @ContractState, x: felt252) -> (felt252, felt252) {
            (x, x * 2)
        }
    }
}
