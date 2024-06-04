#[contract]
mod DivModTestContract {
   #[external]
    fn div_mod_test(x: u8, y: u8) -> u8 {
        let (res, _) = integer::u8_safe_divmod(x, integer::u8_as_non_zero(y));
        res
    }
}
