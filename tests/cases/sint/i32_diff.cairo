fn diff(a: i32, b: i32) -> (u32, u32) {
    match integer::i32_diff(a, b) {
        Result::Ok(r) => (r, 0),
        Result::Err(r) => (r, 1),
    }
}

fn main() -> (
    (u32, u32), (u32, u32),
    (u32, u32), (u32, u32),
    (u32, u32), (u32, u32),
    (u32, u32), (u32, u32),
) {
    (
        diff(18, 1),
        diff(1, 18),
        diff(0, 2147483646),
        diff(2147483646, 0),
        diff(-18, 1),
        diff(1, -18),
        diff(0, -2147483646),
        diff(-2147483646, 0),
    )
}
