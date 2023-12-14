use integer::i16_wide_mul;

fn main() -> (
    i32, i32, i32,
    i32, i32, i32,
    i32, i32, i32,
    i32, i32, i32,
) {
    (
        i16_wide_mul(0_i16, 0_i16),
        i16_wide_mul(0_i16, 10_i16),
        i16_wide_mul(0_i16, 255_i16),
        i16_wide_mul(10_i16, 0_i16),
        i16_wide_mul(10_i16, 10_i16),
        i16_wide_mul(10_i16, 255_i16),
        i16_wide_mul(255_i16, 0_i16),
        i16_wide_mul(255_i16, 10_i16),
        i16_wide_mul(255_i16, 255_i16),
        i16_wide_mul(10_i16, -10_i16),
        i16_wide_mul(10_i16, -255_i16),
        i16_wide_mul(-255_i16, -255_i16),

    )
}
