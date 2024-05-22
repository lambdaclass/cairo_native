use core::nullable::{nullable_from_box, match_nullable, FromNullableResult};


#[derive(Drop, Copy)]
struct FP16x16 {
    mag: u32,
    sign: bool
}

fn main() -> SquashedFelt252Dict<Nullable<FP16x16>> {
    // Create the dictionary
    let mut d: Felt252Dict<Nullable<FP16x16>> = Default::default();

    let box_a = BoxTrait::new(identity(FP16x16 { mag: 1, sign: false }));
    let box_b = BoxTrait::new(identity(FP16x16 { mag: 1, sign: true }));
    let box_c = BoxTrait::new(identity(FP16x16 { mag: 1, sign: true }));

    // Insert it as a `Span`
    d.insert(0, nullable_from_box(box_c));
    d.insert(1, nullable_from_box(box_a));
    d.insert(2, nullable_from_box(box_b));

    d.squash()
}

// TODO: remove this temporary fixed once fixed in cairo
#[inline(never)]
fn identity<T>(t: T) -> T { t }
