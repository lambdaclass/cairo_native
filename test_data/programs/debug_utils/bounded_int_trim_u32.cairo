#[feature("bounded-int-utils")]
use core::internal::bounded_int::{self, BoundedInt};
use core::internal::OptionRev;

fn main() -> BoundedInt<0, 4294967294> {
    let num = match bounded_int::trim_max::<u32>(0xfffffffe) {
        OptionRev::Some(n) => n,
        OptionRev::None => 0,
    };

    num
}
