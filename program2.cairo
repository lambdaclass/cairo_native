use core::option::OptionTrait;
extern type BoundedInt<const MIN: felt252, const MAX: felt252>;
extern fn downcast<T, S>(index: T) -> Option<S> implicits(RangeCheck) nopanic;
extern fn upcast<T, S>(index: T) -> S nopanic;


fn main() -> felt252 {
    let x: BoundedInt<0, 0> = downcast(1).unwrap();
    let value: felt252 = upcast(x);
    // assert(value == 1453543, 'felt should be 0');
    value
}
