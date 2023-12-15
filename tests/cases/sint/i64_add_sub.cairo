use integer::{i64_overflowing_add_impl, i64_overflowing_sub_impl, SignedIntegerResult};

fn overflowing_add(lhs: i64, rhs: i64) -> (i64, i64) {
    match i64_overflowing_add_impl(lhs, rhs) {
        SignedIntegerResult::InRange(res) => (res, 0),
        SignedIntegerResult::Underflow(res) => (res, 1),
        SignedIntegerResult::Overflow(res) =>(res, 2),
    }
}

fn overflowing_sub(lhs: i64, rhs: i64) -> (i64, i64) {
    match i64_overflowing_sub_impl(lhs, rhs) {
        SignedIntegerResult::InRange(res) => (res, 0),
        SignedIntegerResult::Underflow(res) => (res, 1),
        SignedIntegerResult::Overflow(res) =>(res, 2),
    }
}

fn main() -> (
    (i64, i64, i64, i64, i64, i64, i64, i64),
    (i64, i64, i64, i64, i64, i64, i64, i64),
) {
    // In range additions
    let (res_a, flag_a) = overflowing_add(16, 1);
    let (res_b, flag_b) = overflowing_add(-1, -16);
    // Underflow
    let (res_c, flag_c) = overflowing_add(-9000000000000000000, -2000000000000000000);
    // Overflow
    let (res_d, flag_d) = overflowing_add(9000000000000000000, 2000000000000000000);

    // In range subtractions
    let (res_e, flag_e) = overflowing_sub(16, 1);
    let (res_f, flag_f) = overflowing_sub(1, 16);
    // Underflow
    let (res_g, flag_g) = overflowing_sub(-9000000000000000000, 2000000000000000000);
    // Overflow
    let (res_h, flag_h) = overflowing_sub(9000000000000000000, -2000000000000000000);

    (
        ( res_a, flag_a, res_b, flag_b, res_c, flag_c, res_d, flag_d ),
        ( res_e, flag_e, res_f, flag_f, res_g, flag_g, res_h, flag_h ),
    )
}
