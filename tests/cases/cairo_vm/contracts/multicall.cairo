#[starknet::contract]
mod MultiCall {
    // use starknet::account::Call;
    use starknet::ContractAddress;
    use core::debug::PrintTrait;
    use core::integer::u256;
    #[storage]
    struct Storage {}

    #[derive(Drop, Serde)]
    pub struct Call {
        pub to: ContractAddress,
        pub selector: u256,
        pub calldata: Span<felt252>
    }

    #[external(v0)]
    fn fib(self: @ContractState, calls: Span<Call>) -> felt252 {
        let mut calls = calls;
        loop {
            match calls.pop_front() {
                Option::Some(call) => {
                    (*call.to).print();
                    (*call.selector).print();
                    (*(*call.calldata).at(0)).print();
                    (*(*call.calldata).at(1)).print();
                    // (*(call.calldata).at(0)).print();
                    // (*(call.calldata).at(1)).print();
                },
                Option::None => { break; },
            }
        };
        12
    }
}