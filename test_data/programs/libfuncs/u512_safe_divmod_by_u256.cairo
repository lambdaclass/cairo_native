use core::integer::{u512, u512_safe_divmod_by_u256};

fn run_test(lhs: u512, rhs: NonZero<u256>) -> (u512, u256) {
    let (lhs, rhs, _, _, _, _, _) = u512_safe_divmod_by_u256(lhs, rhs);
    (lhs, rhs)
}
