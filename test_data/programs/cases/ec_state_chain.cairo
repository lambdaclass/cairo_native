use core::ec::{EcPointTrait, EcState, EcStateTrait};

/// Accumulates points into an `EcState` and returns the accumulator itself,
/// without finalizing it.
fn main() -> EcState {
    let p = EcPointTrait::new_nz(
        1, 0x27ff039852193be63e77b8e6adc0b6fbe76e54ff6ad53510fdc370d58446a68,
    )
        .unwrap();
    let q = EcPointTrait::new_nz(
        1234, 1301976514684871091717790968549291947487646995000837413367950573852273027507,
    )
        .unwrap();

    let mut state = EcStateTrait::init();
    state.add(p);
    state.add(q);
    state.add_mul(3, p);
    state
}
