use traits::TryInto;

fn main() -> (
    felt252,
    felt252,
    Option<i16>,
    Option<i16>,
    Option<i16>,
) {
    (
        17_i16.into(),
        -17_i16.into(),
        17.try_into(),
        32769.try_into(),
        24857893469346.try_into(),
    )
}
