use core::ec::{ec_neg, EcPoint};

fn run_test(point: EcPoint) -> EcPoint {
    ec_neg(point)
}