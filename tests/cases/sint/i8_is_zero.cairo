use zeroable::IsZeroResult;

fn main() -> bool {
    match integer::i8_is_zero(17) {
        IsZeroResult::Zero => true,
        IsZeroResult::NonZero(_) => false,
    }
}
