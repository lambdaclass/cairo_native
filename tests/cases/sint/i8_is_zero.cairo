use zeroable::IsZeroResult;

fn is_zero(val: i8) -> bool {
    match integer::i8_is_zero(val) {
        IsZeroResult::Zero => true,
        IsZeroResult::NonZero(_) => false,
    }
}
fn main() -> (bool, bool, bool) {
    (
        is_zero(17),
        is_zero(-17),
        is_zero(0),
    )
}
