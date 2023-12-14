use integer::{i16_overflowing_add_impl, SignedIntegerResult};

fn overflowing_add(lhs: i16, rhs: i16) -> (i16, i16) {
    match i16_overflowing_add_impl(lhs, rhs) {
        SignedIntegerResult::InRange(res) => (res, 0),
        SignedIntegerResult::Underflow(res) => (res, 1),
        SignedIntegerResult::Overflow(res) =>(res, 2),
    }
}

fn main() -> (i16, i16, i16, i16, i16, i16, i16, i16) {
    // In range additions
    let (res_a, flag_a) = overflowing_add(16, 1);
    let (res_b, flag_b) = overflowing_add(-1, -16);
    // Underflow
    let (res_c, flag_c) = overflowing_add(-3000, -1000);
    // Overflow
    let (res_d, flag_d) = overflowing_add(1000, 3000);

    ( res_a, flag_a, res_b, flag_b, res_c, flag_c, res_d, flag_d )
}
