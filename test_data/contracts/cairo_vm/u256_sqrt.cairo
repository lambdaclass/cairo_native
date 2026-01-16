#[starknet::contract]
mod U256Sqrt {
    #[storage]
    struct Storage {}

    use core::num::traits::Sqrt;
    use core::traits::Into;
    use traits::TryInto;
    use option::OptionTrait;
    use core::num::traits::Bounded;

    fn as_u256(a: u128, b: u128) -> u256{
        u256{
            low: a,
            high: b
        }
    }

    #[external(v0)]
    fn sqrt(self: @ContractState, num: felt252) -> felt252 {
        let num_in_u128: u128 = num.try_into().unwrap();
        let num_in_u256: u256 = as_u256(num_in_u128, 0);
        let a: u128 = num_in_u256.sqrt();
        let to_return: felt252 = a.into();
        to_return
    }
}
