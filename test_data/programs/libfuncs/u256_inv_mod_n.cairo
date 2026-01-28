use core::math::u256_inv_mod;

fn run_test(a: u256, n: NonZero<u256>) -> Option<NonZero<u256>> {
    u256_inv_mod(a, n)
}
