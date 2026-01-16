use core::ec::{ec_state_try_finalize_nz, EcPoint, EcState};
use core::zeroable::NonZero;

fn run_test(state: EcState) -> Option<NonZero<EcPoint>> {
    ec_state_try_finalize_nz(state)
}