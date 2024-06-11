struct NullableVec<T> {
    items: SquashedFelt252Dict<Nullable<Box<T>>>,
    len: usize,
}

fn main() -> NullableVec<u32> {
    let mut d: Felt252Dict<Nullable<Box<u32>>> = Default::default();

    // Populate the dictionary
    d.insert(0, nullable_from_box(BoxTrait::new(BoxTrait::new(identity(10)))));
    d.insert(1, nullable_from_box(BoxTrait::new(BoxTrait::new(identity(20)))));
    d.insert(2, nullable_from_box(BoxTrait::new(BoxTrait::new(identity(30)))));

    // Return NullableVec
    NullableVec {
        items: d.squash(),
        len: 3,
    }
}

// TODO: remove this temporary fixed once fixed in cairo
#[inline(never)]
fn identity<T>(t: T) -> T { t }
