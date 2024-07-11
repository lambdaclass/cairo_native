#[starknet::contract]
mod WithDraw {
    use starknet::ContractAddress;

    #[storage]
    struct Storage {}

    #[external(v0)]
    fn withdraw(self: @ContractState, token: ContractAddress, amount: felt252) -> bool {
        true
    }
}
