#[feature("bounded-int-utils")]
use core::internal::bounded_int::{self, BoundedInt};
use core::internal::OptionRev;

fn main() -> BoundedInt<-127, 127> {
    let num = match bounded_int::trim_min::<i8>(1) {
        OptionRev::Some(n) => n,
        OptionRev::None => 1,
    };

    num
}
