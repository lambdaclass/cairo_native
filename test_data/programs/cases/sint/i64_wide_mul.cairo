use core::num::traits::WideMul;

fn main() -> (
    i128, i128, i128,
    i128, i128, i128,
    i128, i128, i128,
    i128, i128, i128,
) {
    (
        WideMul::wide_mul(0_i64, 0_i64),
        WideMul::wide_mul(0_i64, 10_i64),
        WideMul::wide_mul(0_i64, 9223372036854775807_i64),
        WideMul::wide_mul(10_i64, 0_i64),
        WideMul::wide_mul(10_i64, 10_i64),
        WideMul::wide_mul(10_i64, 9223372036854775807_i64),
        WideMul::wide_mul(9223372036854775807_i64, 0_i64),
        WideMul::wide_mul(9223372036854775807_i64, 10_i64),
        WideMul::wide_mul(9223372036854775807_i64, 9223372036854775807_i64),
        WideMul::wide_mul(10_i64, -10_i64),
        WideMul::wide_mul(10_i64, -9223372036854775807_i64),
        WideMul::wide_mul(-9223372036854775807_i64, -9223372036854775807_i64),

    )
}
