#[starknet::interface]
pub trait IBlake2sContract<TContractState> {
    
}

#[starknet::contract]
mod Blake2sContract {
    #[storage]
    struct Storage {}

    impl IBlake2sContractImpl of super::IBlake2sContract<ContractState> {}
}
