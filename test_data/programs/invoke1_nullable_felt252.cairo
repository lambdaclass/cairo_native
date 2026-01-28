use core::nullable::{match_nullable, FromNullableResult};

fn main(x: Nullable<felt252>) -> Option<felt252> {
    match match_nullable(x) {
        FromNullableResult::Null(()) => Option::None(()),
        FromNullableResult::NotNull(x) => Option::Some(x.unbox()),
    }
}
