use core::nullable::{match_nullable, FromNullableResult};

fn main(x: Nullable<bool>) -> felt252 {
    match match_nullable(x) {
        FromNullableResult::Null(()) => 100,
        FromNullableResult::NotNull(x) => if x.unbox() {
            1
        } else {
            0
        },
    }
}
