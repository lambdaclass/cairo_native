#[derive(Copy, Drop)]
pub extern type IntRange<T>;
pub extern fn int_range_try_new<T>(
    x: T, y: T
) -> Result<IntRange<T>, IntRange<T>> implicits(core::RangeCheck) nopanic;

fn run_test(lhs: u64, rhs: u64) -> IntRange<u64> {
    int_range_try_new(lhs, rhs).unwrap()
}
