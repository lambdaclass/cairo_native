use core::qm31::m31_ops;
use core::qm31::m31;

fn run_test_1() -> m31 {
    m31_ops::div(25, 5)
}

fn run_test_2() -> m31 {
    m31_ops::div(0x567effa3, 0x567effa9)
}
