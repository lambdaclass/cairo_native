#[feature("bounded-int-utils")]
use core::internal::bounded_int::{self, BoundedInt, is_zero};
use core::zeroable::IsZeroResult;

fn run_test_1(a: felt252) -> bool {
    let bi: BoundedInt<0, 5> = a.try_into().unwrap();
    match is_zero(bi) {
        IsZeroResult::Zero => true,
        IsZeroResult::NonZero(_) => false,
    }
}

fn run_test_2(a: felt252) -> bool {
    let bi: BoundedInt<-5, 5> = a.try_into().unwrap();
    match is_zero(bi) {
        IsZeroResult::Zero => true,
        IsZeroResult::NonZero(_) => false,
    }
}
