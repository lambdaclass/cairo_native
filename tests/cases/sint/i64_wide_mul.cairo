use integer::i64_wide_mul;

fn main() -> (
    i128, i128, i128,
    i128, i128, i128,
    i128, i128, i128,
    i128, i128, i128,
) {
    (
        i64_wide_mul(0_i64, 0_i64),
        i64_wide_mul(0_i64, 10_i64),
        i64_wide_mul(0_i64, 9223372036854775807_i64),
        i64_wide_mul(10_i64, 0_i64),
        i64_wide_mul(10_i64, 10_i64),
        i64_wide_mul(10_i64, 9223372036854775807_i64),
        i64_wide_mul(9223372036854775807_i64, 0_i64),
        i64_wide_mul(9223372036854775807_i64, 10_i64),
        i64_wide_mul(9223372036854775807_i64, 9223372036854775807_i64),
        i64_wide_mul(10_i64, -10_i64),
        i64_wide_mul(10_i64, -9223372036854775807_i64),
        i64_wide_mul(-9223372036854775807_i64, -9223372036854775807_i64),

    )
}
