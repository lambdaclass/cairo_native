use core::qm31::m31_ops;
use core::qm31::m31;

fn run_test_1() -> m31 {
    m31_ops::add(1, 1)
}

fn run_test_2() -> m31 {
    m31_ops::add(0x567effa3, 0x5ffeb970)
}

fn run_test_3() -> m31 {
    m31_ops::add(0x7ffffffe, 1)
}
