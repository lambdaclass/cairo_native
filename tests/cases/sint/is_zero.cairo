use zeroable::IsZeroResult;

fn is_zero_i8(val: i8) -> bool {
    match integer::i8_is_zero(val) {
        IsZeroResult::Zero => true,
        IsZeroResult::NonZero(_) => false,
    }
}

fn is_zero_i16(val: i16) -> bool {
    match integer::i16_is_zero(val) {
        IsZeroResult::Zero => true,
        IsZeroResult::NonZero(_) => false,
    }
}

fn is_zero_i32(val: i32) -> bool {
    match integer::i32_is_zero(val) {
        IsZeroResult::Zero => true,
        IsZeroResult::NonZero(_) => false,
    }
}

fn is_zero_i64(val: i64) -> bool {
    match integer::i64_is_zero(val) {
        IsZeroResult::Zero => true,
        IsZeroResult::NonZero(_) => false,
    }
}

fn main() -> (
    bool, bool, bool,
    bool, bool, bool,
    bool, bool, bool,
    bool, bool, bool,
) {
    (
        is_zero_i8(17),
        is_zero_i8(-17),
        is_zero_i8(0),

        is_zero_i16(17),
        is_zero_i16(-17),
        is_zero_i16(0),

        is_zero_i32(17),
        is_zero_i32(-17),
        is_zero_i32(0),

        is_zero_i64(17),
        is_zero_i64(-17),
        is_zero_i64(0),
    )
}
