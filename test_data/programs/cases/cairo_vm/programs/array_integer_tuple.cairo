use core::array::ArrayTrait;


fn main() -> (Array<u32>, u32) {
    let mut numbers = ArrayTrait::new();
    numbers.append(1);

    (numbers, 1)
}
