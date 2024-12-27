use core::internal::OptionRev;
use core::internal::bounded_int;
fn main() {
    assert!(bounded_int::trim::<u8, 0>(1) == OptionRev::Some(1));
}
