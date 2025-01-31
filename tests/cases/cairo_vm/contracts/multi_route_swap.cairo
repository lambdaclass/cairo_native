#[starknet::contract]
mod MultiCall {
    use starknet::ContractAddress;
    use core::debug::PrintTrait;
    use core::integer::u256;
    #[storage]
    struct Storage {}

    #[derive(Drop, Serde)]
    pub struct Route {
        pub to: ContractAddress,
        pub selector: u256,
        pub calldata: Span<felt252>
    }

    #[external(v0)]
    fn multi_route_swap(
        self: @ContractState,
        token_from_address: ContractAddress,
        token_from_amount: u256,
        token_to_address: ContractAddress,
        token_to_amount: u256,
        token_to_min_amount: u256,
        beneficiary: ContractAddress,
        integrator_fee_amount_bps: u128,
        integrator_fee_recipient: ContractAddress,
        routes: Span<Route>,
    ) -> bool {
        true
    }
}
