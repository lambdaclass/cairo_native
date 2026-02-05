use core::ec::{ec_state_add, EcPoint, EcState};
use core::zeroable::NonZero;

fn run_test(mut state: EcState, point: NonZero<EcPoint>) -> EcState {
    ec_state_add(ref state, point);
    state
}
