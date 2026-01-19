use core::qm31::{QM31Trait, qm31, m31, qm31_from_m31};

fn run_test_with_0() -> qm31 {
    qm31_from_m31(0)
}

fn run_test_with_1() -> qm31 {
    qm31_from_m31(1)
}

fn run_test_with_big_coefficient() -> qm31 {
    qm31_from_m31(0x60713d44)
}
