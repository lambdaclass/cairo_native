use core::ec::{ec_neg_nz, NonZeroEcPoint};

fn run_test(x: NonZeroEcPoint) -> NonZeroEcPoint {
    ec_neg_nz(x)
}
