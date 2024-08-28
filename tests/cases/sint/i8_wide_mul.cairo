use core::num::traits::WideMul;

fn main() -> (
    i16, i16, i16,
    i16, i16, i16,
    i16, i16, i16,
    i16, i16, i16,
) {
    (
        WideMul::wide_mul(0_i8, 0_i8),
        WideMul::wide_mul(0_i8, 10_i8),
        WideMul::wide_mul(0_i8, 127_i8),
        WideMul::wide_mul(10_i8, 0_i8),
        WideMul::wide_mul(10_i8, 10_i8),
        WideMul::wide_mul(10_i8, 127_i8),
        WideMul::wide_mul(127_i8, 0_i8),
        WideMul::wide_mul(127_i8, 10_i8),
        WideMul::wide_mul(127_i8, 127_i8),
        WideMul::wide_mul(10_i8, -10_i8),
        WideMul::wide_mul(10_i8, -127_i8),
        WideMul::wide_mul(-127_i8, -127_i8),

    )
}
