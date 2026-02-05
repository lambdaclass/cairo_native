// FP16x16
#[derive(Serde, Copy, Drop)]
struct FP16x16 {
    mag: u32,
    sign: bool
}

trait FixedTrait<T, MAG> {
    fn new(mag: MAG, sign: bool) -> T;
}

impl FP16x16Impl of FixedTrait<FP16x16, u32> {
    fn new(mag: u32, sign: bool) -> FP16x16 {
        FP16x16 { mag: mag, sign: sign }
    }
}

//Tensor
#[derive(Copy, Drop)]
struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>,
}

trait TensorTrait<T> {
    fn new(shape: Span<usize>, data: Span<T>) -> Tensor<T>;
}

impl FP16x16Tensor of TensorTrait<FP16x16> {
    fn new(shape: Span<usize>, data: Span<FP16x16>) -> Tensor<FP16x16> {
        new_tensor(shape, data)
    }
}

fn new_tensor<T>(shape: Span<usize>, data: Span<T>) -> Tensor<T> {
    check_shape::<T>(shape, data);
    Tensor::<T> { shape, data }
}

fn check_shape<T>(shape: Span<usize>, data: Span<T>) {
    assert(len_from_shape(shape) == data.len(), 'wrong tensor shape');
}

fn len_from_shape(mut shape: Span<usize>) -> usize {
    let mut result: usize = 1;

    loop {
        match shape.pop_front() {
            Option::Some(item) => { result *= *item; },
            Option::None => { break; }
        };
    };

    result
}

fn main() -> Tensor<FP16x16> {
     TensorTrait::new(
         array![1, 2].span(),
         array![
           FixedTrait::new(1, false), 
           FixedTrait::new(1, true)
         ].span()
    )
}
