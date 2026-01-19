use core::qm31::{QM31Trait, m31, qm31};

fn run_test_1() -> [m31;4] {
    let qm31 = QM31Trait::new(1, 2, 3, 4);
    let unpacked_qm31 = qm31.unpack();

    unpacked_qm31
}

fn run_test_2() -> [m31;4] {
    let qm31 = QM31Trait::new(0x544b2fba, 0x673cff77, 0x60713d44, 0x499602d2);
    let unpacked_qm31 = qm31.unpack();

    unpacked_qm31
}
