#[starknet::contract]
mod Echo {
    #[storage]
    struct Storage {
        balance: felt252,
    }

    #[constructor]
    fn constructor(ref self: ContractState, initial_balance: felt252) {
        panic_with_felt252('panic');
        //self.balance.write(initial_balance);
    }

    #[external(v0)]
    fn echo(ref self: ContractState, value: felt252) -> felt252 {
        value
    }
}
