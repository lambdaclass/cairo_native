fn main() -> (
    bool, bool, bool,
    bool, bool, bool,
    bool, bool, bool,
    bool, bool, bool,
) {
    (
        integer::i8_eq(17, 71),
        integer::i8_eq(17, 17),
        integer::i8_eq(2, -2),

        integer::i16_eq(17, 71),
        integer::i16_eq(17, 17),
        integer::i16_eq(2, -2),

        integer::i32_eq(17, 71),
        integer::i32_eq(17, 17),
        integer::i32_eq(2, -2),

        integer::i64_eq(17, 71),
        integer::i64_eq(17, 17),
        integer::i64_eq(2, -2),
    )
}
