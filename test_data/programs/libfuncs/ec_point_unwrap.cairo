use core::{ec::{ec_point_unwrap, EcPoint}, zeroable::NonZero};

fn run_test(point: NonZero<EcPoint>) -> (felt252, felt252) {
    ec_point_unwrap(point)
}