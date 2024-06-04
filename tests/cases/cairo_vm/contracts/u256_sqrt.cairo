#[contract]
mod U256Sqrt {
    use integer::u256_sqrt;
    use core::traits::Into;
    use traits::TryInto;
    use option::OptionTrait;    
    use integer::BoundedInt;

    #[internal]
    fn as_u256(a: u128, b: u128) -> u256{
        u256{
            low: a,
            high: b
        }
    }
    #[external]
    fn sqrt(num: felt252) -> felt252 {
        let num_in_u128: u128 = num.try_into().unwrap();
        let num_in_u256: u256 = as_u256(num_in_u128, 0);
        let a: u128 = u256_sqrt(num_in_u256);
        let to_return: felt252 = a.into();
        to_return
    }

    #[external]
    fn sqrt_big_num() -> felt252 {
        let num: u128 = 1267650600228229401496703205376_u128;
        let a: u128 = u256_sqrt(as_u256(num, 0));
        let to_return: felt252 = a.into();
        to_return
    }

    #[external]
    fn sqrt_max_num() -> felt252 {
        let a: u128 = u256_sqrt(BoundedInt::max());
        if a != BoundedInt::max() {
            0
        }else{
            1
        }
    }
}
