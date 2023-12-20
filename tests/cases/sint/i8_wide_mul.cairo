use integer::i8_wide_mul;

fn main() -> (
    i16, i16, i16,
    i16, i16, i16,
    i16, i16, i16,
    i16, i16, i16,
) {
    (
        i8_wide_mul(0_i8, 0_i8),
        i8_wide_mul(0_i8, 10_i8),
        i8_wide_mul(0_i8, 127_i8),
        i8_wide_mul(10_i8, 0_i8),
        i8_wide_mul(10_i8, 10_i8),
        i8_wide_mul(10_i8, 127_i8),
        i8_wide_mul(127_i8, 0_i8),
        i8_wide_mul(127_i8, 10_i8),
        i8_wide_mul(127_i8, 127_i8),
        i8_wide_mul(10_i8, -10_i8),
        i8_wide_mul(10_i8, -127_i8),
        i8_wide_mul(-127_i8, -127_i8),

    )
}
