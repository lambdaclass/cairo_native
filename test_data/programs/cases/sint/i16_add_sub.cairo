use integer::{i16_overflowing_add_impl, i16_overflowing_sub_impl, SignedIntegerResult};

fn overflowing_add(lhs: i16, rhs: i16) -> (i16, i16) {
    match i16_overflowing_add_impl(lhs, rhs) {
        SignedIntegerResult::InRange(res) => (res, 0),
        SignedIntegerResult::Underflow(res) => (res, 1),
        SignedIntegerResult::Overflow(res) =>(res, 2),
    }
}

fn overflowing_sub(lhs: i16, rhs: i16) -> (i16, i16) {
    match i16_overflowing_sub_impl(lhs, rhs) {
        SignedIntegerResult::InRange(res) => (res, 0),
        SignedIntegerResult::Underflow(res) => (res, 1),
        SignedIntegerResult::Overflow(res) =>(res, 2),
    }
}

fn main() -> (
    (i16, i16, i16, i16, i16, i16, i16, i16),
    (i16, i16, i16, i16, i16, i16, i16, i16),
) {
    // In range additions
    let (res_a, flag_a) = overflowing_add(16, 1);
    let (res_b, flag_b) = overflowing_add(-1, -16);
    // Underflow
    let (res_c, flag_c) = overflowing_add(-3000, -1000);
    // Overflow
    let (res_d, flag_d) = overflowing_add(1000, 3000);

    // In range subtractions
    let (res_e, flag_e) = overflowing_sub(16, 1);
    let (res_f, flag_f) = overflowing_sub(1, 16);
    // Underflow
    let (res_g, flag_g) = overflowing_sub(-3000, 1000);
    // Overflow
    let (res_h, flag_h) = overflowing_sub(1000, -3000);

    (
        ( res_a, flag_a, res_b, flag_b, res_c, flag_c, res_d, flag_d ),
        ( res_e, flag_e, res_f, flag_f, res_g, flag_g, res_h, flag_h ),
    )
}
