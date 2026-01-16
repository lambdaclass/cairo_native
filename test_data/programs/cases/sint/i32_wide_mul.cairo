use core::num::traits::WideMul;

fn main() -> (
    i64, i64, i64,
    i64, i64, i64,
    i64, i64, i64,
    i64, i64, i64,
) {
    (
        WideMul::wide_mul(0_i32, 0_i32),
        WideMul::wide_mul(0_i32, 10_i32),
        WideMul::wide_mul(0_i32, 2147483647_i32),
        WideMul::wide_mul(10_i32, 0_i32),
        WideMul::wide_mul(10_i32, 10_i32),
        WideMul::wide_mul(10_i32, 2147483647_i32),
        WideMul::wide_mul(2147483647_i32, 0_i32),
        WideMul::wide_mul(2147483647_i32, 10_i32),
        WideMul::wide_mul(2147483647_i32, 2147483647_i32),
        WideMul::wide_mul(10_i32, -10_i32),
        WideMul::wide_mul(10_i32, -2147483647_i32),
        WideMul::wide_mul(-2147483647_i32, -2147483647_i32),

    )
}
