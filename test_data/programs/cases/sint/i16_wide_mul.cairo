use core::num::traits::WideMul;

fn main() -> (
    i32, i32, i32,
    i32, i32, i32,
    i32, i32, i32,
    i32, i32, i32,
) {
    (
        WideMul::wide_mul(0_i16, 0_i16),
        WideMul::wide_mul(0_i16, 10_i16),
        WideMul::wide_mul(0_i16, 255_i16),
        WideMul::wide_mul(10_i16, 0_i16),
        WideMul::wide_mul(10_i16, 10_i16),
        WideMul::wide_mul(10_i16, 255_i16),
        WideMul::wide_mul(255_i16, 0_i16),
        WideMul::wide_mul(255_i16, 10_i16),
        WideMul::wide_mul(255_i16, 255_i16),
        WideMul::wide_mul(10_i16, -10_i16),
        WideMul::wide_mul(10_i16, -255_i16),
        WideMul::wide_mul(-255_i16, -255_i16),

    )
}
