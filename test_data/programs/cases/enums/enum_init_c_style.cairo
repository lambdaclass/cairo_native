enum CEnum {
    A: (),
    B: (),
    C: (),
}

fn main() -> (CEnum, CEnum, CEnum) {
    (
        CEnum::A(()),
        CEnum::B(()),
        CEnum::C(()),
    )
}
