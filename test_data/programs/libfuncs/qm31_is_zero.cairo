use core::qm31::{QM31Trait, qm31, qm31_is_zero};
use core::internal::OptionRev;

fn run_test(input: qm31) -> OptionRev<NonZero<qm31>> {
    qm31_is_zero(input)
}

fn run_test_edge_case() -> OptionRev<NonZero<qm31>> {
    let lhs = QM31Trait::new(0x7ffffffe, 0x7ffffffe, 0x7ffffffe, 0x7ffffffe);
    let rhs = QM31Trait::new(1, 1, 1, 1);
    let qm31 = lhs + rhs;
    qm31_is_zero(qm31)
}
