use box::BoxTrait;
use box::BoxImpl;

fn run_test() -> u32 {
    let x: u32 = 2_u32;
    let box_x: Box<u32> = BoxTrait::new(x);
    box_x.unbox()
}