fn diff(a: i8, b: i8) -> (u8, u8) {
    match integer::i8_diff(a, b) {
        Result::Ok(r) => (r, 0),
        Result::Err(r) => (r, 1),
    }
}

fn main() -> (
    (u8, u8), (u8, u8),
    (u8, u8), (u8, u8),
    (u8, u8), (u8, u8),
    (u8, u8), (u8, u8),
) {
    (
        diff(18, 1),
        diff(1, 18),
        diff(0, 127),
        diff(127, 0),
        diff(-18, 1),
        diff(1, -18),
        diff(0, -127),
        diff(-127, 0),
    )
}
