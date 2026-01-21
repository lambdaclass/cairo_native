#[starknet::contract]
mod TestLessThan {
    #[storage]
    struct Storage {}

    use integer::upcast;
    use integer::downcast;
    use option::OptionTrait;

    // tests whether the input (u128) can be downcast to an u8
    #[external(v0)]
    fn test_less_than_with_downcast(self: @ContractState, number: u128) -> bool {
        let downcast_test: Option<u8> = downcast(number);

        match downcast_test {
            Option::Some(_) => { return true; },
            Option::None(_) => { return false; }
        }
    }
}
