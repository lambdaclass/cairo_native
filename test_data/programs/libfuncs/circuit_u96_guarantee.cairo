use core::circuit::{into_u96_guarantee, U96Guarantee};
#[feature("bounded-int-utils")]
use core::internal::bounded_int::BoundedInt;

fn main() -> (U96Guarantee, U96Guarantee, U96Guarantee) {
    (
        into_u96_guarantee::<BoundedInt<0, 79228162514264337593543950335>>(123),
        into_u96_guarantee::<BoundedInt<100, 1000>>(123),
        into_u96_guarantee::<u8>(123),
    )
}