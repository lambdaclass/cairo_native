use core::nullable::{nullable_from_box, match_nullable, FromNullableResult};

fn main() -> Nullable<Span<felt252>> {
   let a = array![8, 9, 10, 11];
   nullable_from_box(BoxTrait::new(a.span()))
}
