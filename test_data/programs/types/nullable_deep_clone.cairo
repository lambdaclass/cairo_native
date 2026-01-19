use core::array::ArrayTrait;
use core::NullableTrait;

fn run_test() -> @Nullable<Array<felt252>> {
    let mut x = NullableTrait::new(array![1, 2, 3]);
    let x_s = @x;

    let mut y = NullableTrait::deref(x);
    y.append(4);

    x_s
}
