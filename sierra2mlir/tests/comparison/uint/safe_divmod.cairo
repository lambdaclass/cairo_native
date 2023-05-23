use integer::u8_try_as_non_zero;
use integer::u8_safe_divmod;
use integer::u16_try_as_non_zero;
use integer::u16_safe_divmod;
use integer::u32_try_as_non_zero;
use integer::u32_safe_divmod;
use integer::u64_try_as_non_zero;
use integer::u64_safe_divmod;
use integer::u128_try_as_non_zero;
use integer::u128_safe_divmod;
use integer::u256_try_as_non_zero;
use integer::u256_safe_divmod;

fn main() -> (
    (u8, u8, bool),
    (u8, u8, bool),
    (u8, u8, bool),
    (u8, u8, bool),
    (u8, u8, bool),
    (u8, u8, bool),
    (u8, u8, bool),
    (u8, u8, bool),
    (u8, u8, bool),

    (u16, u16, bool),
    (u16, u16, bool),
    (u16, u16, bool),
    (u16, u16, bool),
    (u16, u16, bool),
    (u16, u16, bool),
    (u16, u16, bool),
    (u16, u16, bool),
    (u16, u16, bool),

    (u32, u32, bool),
    (u32, u32, bool),
    (u32, u32, bool),
    (u32, u32, bool),
    (u32, u32, bool),
    (u32, u32, bool),
    (u32, u32, bool),
    (u32, u32, bool),
    (u32, u32, bool),

    (u64, u64, bool),
    (u64, u64, bool),
    (u64, u64, bool),
    (u64, u64, bool),
    (u64, u64, bool),
    (u64, u64, bool),
    (u64, u64, bool),
    (u64, u64, bool),
    (u64, u64, bool),

    (u128, u128, bool),
    (u128, u128, bool),
    (u128, u128, bool),
    (u128, u128, bool),
    (u128, u128, bool),
    (u128, u128, bool),
    (u128, u128, bool),
    (u128, u128, bool),
    (u128, u128, bool),

    (u256, u256, bool),
    (u256, u256, bool),
    (u256, u256, bool),
    (u256, u256, bool),
    (u256, u256, bool),
    (u256, u256, bool),
    (u256, u256, bool),
    (u256, u256, bool),
    (u256, u256, bool),
) {
    (
        u8_divmod(0_u8, 0_u8),
        u8_divmod(0_u8, 10_u8),
        u8_divmod(0_u8, 255_u8),
        u8_divmod(10_u8, 0_u8),
        u8_divmod(10_u8, 10_u8),
        u8_divmod(10_u8, 255_u8),
        u8_divmod(255_u8, 0_u8),
        u8_divmod(255_u8, 10_u8),
        u8_divmod(255_u8, 255_u8),

        u16_divmod(0_u16, 0_u16),
        u16_divmod(0_u16, 1000_u16),
        u16_divmod(0_u16, 65535_u16),
        u16_divmod(1000_u16, 0_u16),
        u16_divmod(1000_u16, 1000_u16),
        u16_divmod(1000_u16, 65535_u16),
        u16_divmod(65535_u16, 0_u16),
        u16_divmod(65535_u16, 10_u16),
        u16_divmod(65535_u16, 65535_u16),

        u32_divmod(0_u32, 0_u32),
        u32_divmod(0_u32, 100000_u32),
        u32_divmod(0_u32, 4294967295_u32),
        u32_divmod(100000_u32, 0_u32),
        u32_divmod(100000_u32, 100000_u32),
        u32_divmod(100000_u32, 4294967295_u32),
        u32_divmod(4294967295_u32, 0_u32),
        u32_divmod(4294967295_u32, 10_u32),
        u32_divmod(4294967295_u32, 4294967295_u32),

        u64_divmod(0_u64, 0_u64),
        u64_divmod(0_u64, 10000000000_u64),
        u64_divmod(0_u64, 18446744073709551615_u64),
        u64_divmod(10000000000_u64, 0_u64),
        u64_divmod(10000000000_u64, 10000000000_u64),
        u64_divmod(10000000000_u64, 18446744073709551615_u64),
        u64_divmod(18446744073709551615_u64, 0_u64),
        u64_divmod(18446744073709551615_u64, 10_u64),
        u64_divmod(18446744073709551615_u64, 18446744073709551615_u64),

        u128_divmod(0_u128, 0_u128),
        u128_divmod(0_u128, 100000000000000000000_u128),
        u128_divmod(0_u128, 340282366920938463463374607431768211455_u128),
        u128_divmod(100000000000000000000_u128, 0_u128),
        u128_divmod(100000000000000000000_u128, 100000000000000000000_u128),
        u128_divmod(100000000000000000000_u128, 340282366920938463463374607431768211455_u128),
        u128_divmod(340282366920938463463374607431768211455_u128, 0_u128),
        u128_divmod(340282366920938463463374607431768211455_u128, 10_u128),
        u128_divmod(340282366920938463463374607431768211455_u128, 340282366920938463463374607431768211455_u128),

        u256_divmod(0_u256, 0_u256),
        u256_divmod(0_u256, 100000000000000000000_u256),
        u256_divmod(0_u256, 115792089237316195423570985008687907853269984665640564039457584007913129639935_u256),
        u256_divmod(100000000000000000000_u256, 0_u256),
        u256_divmod(100000000000000000000_u256, 100000000000000000000_u256),
        u256_divmod(100000000000000000000_u256, 115792089237316195423570985008687907853269984665640564039457584007913129639935_u256),
        u256_divmod(115792089237316195423570985008687907853269984665640564039457584007913129639935_u256, 0_u256),
        u256_divmod(115792089237316195423570985008687907853269984665640564039457584007913129639935_u256, 10_u256),
        u256_divmod(115792089237316195423570985008687907853269984665640564039457584007913129639935_u256, 115792089237316195423570985008687907853269984665640564039457584007913129639935_u256),
    )
}

fn u8_divmod(a: u8, b: u8) -> (u8, u8, bool) {
    let rhs = u8_try_as_non_zero(b);
    match rhs {
        Option::Some(x) => {
            let (res_l, res_r) = u8_safe_divmod(a, x);
            (res_l, res_r, true)
        },
        Option::None(()) => (0_u8, 0_u8, false),
    }
}

fn u16_divmod(a: u16, b: u16) -> (u16, u16, bool) {
    let rhs = u16_try_as_non_zero(b);
    match rhs {
        Option::Some(x) => {
            let (res_l, res_r) = u16_safe_divmod(a, x);
            (res_l, res_r, true)
        },
        Option::None(()) => (0_u16, 0_u16, false),
    }
}

fn u32_divmod(a: u32, b: u32) -> (u32, u32, bool) {
    let rhs = u32_try_as_non_zero(b);
    match rhs {
        Option::Some(x) => {
            let (res_l, res_r) = u32_safe_divmod(a, x);
            (res_l, res_r, true)
        },
        Option::None(()) => (0_u32, 0_u32, false),
    }
}

fn u64_divmod(a: u64, b: u64) -> (u64, u64, bool) {
    let rhs = u64_try_as_non_zero(b);
    match rhs {
        Option::Some(x) => {
            let (res_l, res_r) = u64_safe_divmod(a, x);
            (res_l, res_r, true)
        },
        Option::None(()) => (0_u64, 0_u64, false),
    }
}

fn u128_divmod(a: u128, b: u128) -> (u128, u128, bool) {
    let rhs = u128_try_as_non_zero(b);
    match rhs {
        Option::Some(x) => {
            let (res_l, res_r) = u128_safe_divmod(a, x);
            (res_l, res_r, true)
        },
        Option::None(()) => (0_u128, 0_u128, false),
    }
}

fn u256_divmod(a: u256, b: u256) -> (u256, u256, bool) {
    let rhs = u256_try_as_non_zero(b);
    match rhs {
        Option::Some(x) => {
            let (res_l, res_r) = u256_safe_divmod(a, x);
            (res_l, res_r, true)
        },
        Option::None(()) => (0_u256, 0_u256, false),
    }
}
