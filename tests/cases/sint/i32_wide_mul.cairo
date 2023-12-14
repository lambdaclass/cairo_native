use integer::i32_wide_mul;

fn main() -> (
    i64, i64, i64,
    i64, i64, i64,
    i64, i64, i64,
    i64, i64, i64,
) {
    (
        i32_wide_mul(0_i32, 0_i32),
        i32_wide_mul(0_i32, 10_i32),
        i32_wide_mul(0_i32, 2147483647_i32),
        i32_wide_mul(10_i32, 0_i32),
        i32_wide_mul(10_i32, 10_i32),
        i32_wide_mul(10_i32, 2147483647_i32),
        i32_wide_mul(2147483647_i32, 0_i32),
        i32_wide_mul(2147483647_i32, 10_i32),
        i32_wide_mul(2147483647_i32, 2147483647_i32),
        i32_wide_mul(10_i32, -10_i32),
        i32_wide_mul(10_i32, -2147483647_i32),
        i32_wide_mul(-2147483647_i32, -2147483647_i32),

    )
}
