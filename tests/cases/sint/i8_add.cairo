use integer::{i8_overflowing_add_impl, SignedIntegerResult};

fn overflowing_add(lhs: i8, rhs: i8) -> (i8, i8) {
    match i8_overflowing_add_impl(lhs, rhs) {
        SignedIntegerResult::InRange(res) => (res, 0),
        SignedIntegerResult::Underflow(res) => (res, 1),
        SignedIntegerResult::Overflow(res) =>(res, 2),
    }
}

fn main() -> (i8, i8, i8, i8, i8, i8, i8, i8) {
    // In range additions
    let (res_a, flag_a) = overflowing_add(16, 1);
    let (res_b, flag_b) = overflowing_add(-1, -16);
    // Underflow
    let (res_c, flag_c) = overflowing_add(-100, -100);
    // Overflow
    let (res_d, flag_d) = overflowing_add(100, 100);

    ( res_a, flag_a, res_b, flag_b, res_c, flag_c, res_d, flag_d )
}
