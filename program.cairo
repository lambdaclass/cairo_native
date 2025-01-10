use core::internal::{OptionRev, bounded_int::BoundedInt};
use core::internal::bounded_int;
fn main() -> BoundedInt<-32767, 32767> {
    let num = match bounded_int::trim::<i16, -0x8000>(-0x8000) {
        OptionRev::Some(n) => n,
        OptionRev::None => 0,
    };

    num
}
