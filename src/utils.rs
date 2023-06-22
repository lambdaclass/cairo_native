use cairo_lang_sierra::ids::FunctionId;
use std::{alloc::Layout, borrow::Cow};

pub fn generate_function_name(function_id: &FunctionId) -> Cow<str> {
    function_id
        .debug_name
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(format!("f{}", function_id.id)))
}

pub fn get_integer_layout(width: u32) -> Layout {
    if width == 0 {
        Layout::from_size_align(0, 1).unwrap()
    } else if width <= 8 {
        Layout::from_size_align(1, 1).unwrap()
    } else if width <= 16 {
        Layout::from_size_align(2, 2).unwrap()
    } else if width <= 32 {
        Layout::from_size_align(4, 4).unwrap()
    } else {
        Layout::from_size_align(width.next_multiple_of(8) as usize >> 3, 8).unwrap()
    }
}
