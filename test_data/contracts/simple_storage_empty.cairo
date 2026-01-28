#[starknet::interface]
trait ISimpleStorage<TContractState> {
    fn call(self: @TContractState);
}

#[starknet::contract]
mod contract {
    #[storage]
    struct Storage {}

    #[abi(embed_v0)]
    impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
        fn call(self: @ContractState) {}
    }
}
