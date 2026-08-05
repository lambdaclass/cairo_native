#[feature("bounded-int-utils")]
use core::internal::bounded_int::BoundedInt;

extern const fn downcast<FromType, ToType>(
    x: FromType,
) -> Option<ToType> implicits(RangeCheck) nopanic;

// Downcasts a felt252 into a bounded int whose range [1 - PRIME, 5 - PRIME] is
// strictly negative, so every accepted felt is interpreted as `felt - PRIME`.
fn run_test(
    v: felt252,
) -> Option<
    BoundedInt<
        -3618502788666131213697322783095070105623107215331596699973092056135872020480,
        -3618502788666131213697322783095070105623107215331596699973092056135872020476,
    >,
> {
    downcast(v)
}
