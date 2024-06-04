#[contract]
mod U16Sqrt {
    use integer::u16_sqrt;
    use core::traits::Into;
    use traits::TryInto;
    use option::OptionTrait;    


    #[external]
    fn sqrt(num: felt252) -> felt252 {
        let num_in_u16: u16 = num.try_into().unwrap();
        let a: u8 = u16_sqrt(num_in_u16);
        let to_return: felt252 = a.into();
        to_return
    }
}
