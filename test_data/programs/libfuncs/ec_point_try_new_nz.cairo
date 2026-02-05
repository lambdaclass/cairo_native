use core::ec::{ec_point_try_new_nz, EcPoint};
use core::zeroable::NonZero;

fn run_test(x: felt252, y: felt252) -> Option<NonZero<EcPoint>> {
    ec_point_try_new_nz(x, y)
}
