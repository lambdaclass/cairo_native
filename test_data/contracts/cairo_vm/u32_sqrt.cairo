#[starknet::contract]
mod U32Sqrt {
    #[storage]
    struct Storage {}

    use core::num::traits::Sqrt;
    use core::traits::Into;
    use traits::TryInto;
    use option::OptionTrait;    


    #[external(v0)]
    fn sqrt(self: @ContractState, num: felt252) -> felt252 {
        let num_in_u32: u32 = num.try_into().unwrap();
        let a: u16 = num_in_u32.sqrt();
        let to_return: felt252 = a.into();
        to_return
    }
}
