use core::{ec::{ec_point_from_x_nz, EcPoint}};
use core::zeroable::NonZero;

fn run_test(x: felt252) -> Option<NonZero<EcPoint>> {
    ec_point_from_x_nz(x)
}
