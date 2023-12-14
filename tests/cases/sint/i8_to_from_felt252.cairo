use traits::TryInto;

fn main() -> (
    felt252,
    felt252,
    Option<i8>,
    Option<i8>,
    Option<i8>,
) {
    (
        17_i8.into(),
        -17_i8.into(),
        17.try_into(),
        150.try_into(),
        24857893469346.try_into(),
    )
}
