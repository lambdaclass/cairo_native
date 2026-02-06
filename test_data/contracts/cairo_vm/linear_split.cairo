#[starknet::contract]
mod LinearSplit {
    #[storage]
    struct Storage {}

    use integer::u16_try_from_felt252;

    #[external(v0)]
    fn cast(self: @ContractState, a: felt252) -> Option<u16> {
        u16_try_from_felt252(a)
    }
}
