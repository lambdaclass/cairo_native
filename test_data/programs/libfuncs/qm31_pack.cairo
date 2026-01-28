use core::qm31::{QM31Trait, qm31};

fn run_test() -> qm31 {
    let qm31 = QM31Trait::new(1, 2, 3, 4);
    qm31
}

fn run_test_large_coefficients() -> qm31 {
    let qm31 = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
    qm31
}
