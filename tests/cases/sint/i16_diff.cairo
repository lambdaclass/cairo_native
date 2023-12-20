fn diff(a: i16, b: i16) -> (u16, u16) {
    match integer::i16_diff(a, b) {
        Result::Ok(r) => (r, 0),
        Result::Err(r) => (r, 1),
    }
}

fn main() -> (
    (u16, u16), (u16, u16),
    (u16, u16), (u16, u16),
    (u16, u16), (u16, u16),
    (u16, u16), (u16, u16),
) {
    (
        diff(18, 1),
        diff(1, 18),
        diff(0, 32767),
        diff(32767, 0),
        diff(-18, 1),
        diff(1, -18),
        diff(0, -32767),
        diff(-32767, 0),
    )
}
