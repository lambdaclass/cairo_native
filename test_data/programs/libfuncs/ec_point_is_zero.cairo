use core::{ec::{ec_point_is_zero, EcPoint}, zeroable::IsZeroResult};

fn run_test(point: EcPoint) -> IsZeroResult<EcPoint> {
    ec_point_is_zero(point)
}
