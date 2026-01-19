use core::qm31::m31_ops;
use core::qm31::m31;

fn run_test_1() -> m31 {
    m31_ops::sub(2, 1)
}

fn run_test_2() -> m31 {
    m31_ops::sub(0x567effa3, 0x567effa9)
}

fn run_test_3() -> m31 {
    m31_ops::sub(0, 1)
}
