use core::array::SpanTrait;
use core::array::ArrayTrait;


fn main() -> Array<Array<u32>> {
    let mut numbers = ArrayTrait::new();
    let mut numbers_a = ArrayTrait::new();
    let mut numbers_b = ArrayTrait::new();
    numbers_a.append(4_u32);
    numbers_a.append(3_u32);
    numbers_b.append(2_u32);
    numbers_b.append(1_u32);
    numbers.append(numbers_a);
    numbers.append(numbers_b);

    numbers
}
