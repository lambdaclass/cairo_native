#[contract]
mod U128Sqrt {
    use integer::u128_sqrt;
    use core::traits::Into;
    use traits::TryInto;
    use option::OptionTrait;    


    #[external]
    fn sqrt(num: felt252) -> felt252 {
        let num_in_u128: u128 = num.try_into().unwrap();
        let a: u64 = u128_sqrt(num_in_u128);
        let to_return: felt252 = a.into();
        to_return
    }
}
