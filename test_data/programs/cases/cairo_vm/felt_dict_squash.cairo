use core::nullable::{nullable_from_box, match_nullable, FromNullableResult};
use core::dict::Felt252DictEntry;

fn main() -> SquashedFelt252Dict<Nullable<Span<felt252>>> {
    // Create the dictionary
    let mut d: Felt252Dict<Nullable<Span<felt252>>> = Default::default();

    // Create the array to insert
    let a = array![8, 9, 10, 11];
    let b = array![1, 2, 3, 44];
    let c = array![4, 5, 6];

    // Insert it as a `Span`
    d.insert(66675, nullable_from_box(BoxTrait::new(a.span())));
    d.insert(66676, nullable_from_box(BoxTrait::new(b.span())));
    d.insert(66675, nullable_from_box(BoxTrait::new(c.span())));
    d.squash()
}
