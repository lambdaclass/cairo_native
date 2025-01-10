use core::ec::{ec_state_add_mul, EcPoint, EcState};
use core::zeroable::NonZero;

fn run_test(mut state: EcState, scalar: felt252, point: NonZero<EcPoint>) -> EcState {
    ec_state_add_mul(ref state, scalar, point);
    state
}
