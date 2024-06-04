#[contract]
mod WideMul128TestContract {

   #[external]
    fn wide_mul_128_test(a: u128, b: u128) -> bool {
        let (res_high, res_low, _) = integer::u128_guarantee_mul(a, b);
        // verify that: `a * b = 2**128 * res_high + res_low`
        let a_256: u256 = integer::u256 {low: a, high: 0};
        let b_256: u256 = integer::u256 {low: b, high: 0};
        let res_high_256: u256 = integer::u256 {low: res_high, high: 0};
        let res_low_256: u256 = integer::u256 {low: res_low, high: 0};
        a_256 * b_256 == 0x100000000000000000000000000000000_u256 * res_high_256 + res_low_256
    }
}
