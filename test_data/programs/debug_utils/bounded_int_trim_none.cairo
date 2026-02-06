#[feature("bounded-int-utils")]
use core::internal::bounded_int::{self, BoundedInt};
use core::internal::OptionRev;

fn main() -> BoundedInt<-32767, 32767> {
    let num = match bounded_int::trim_min::<i16>(-0x8000) {
        OptionRev::Some(n) => n,
        OptionRev::None => 0,
    };

    num
}
