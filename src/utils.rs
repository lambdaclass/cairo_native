//! # Various utilities

use cairo_lang_sierra::ids::FunctionId;
use std::{alloc::Layout, borrow::Cow, fmt};

/// Generate a function name.
///
/// If the program includes function identifiers, return those. Otherwise return `f` followed by the
/// identifier number.
pub fn generate_function_name(function_id: &FunctionId) -> Cow<str> {
    function_id
        .debug_name
        .as_deref()
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(format!("f{}", function_id.id)))
}

/// Return the layout for an integer of arbitrary width.
///
/// This assumes the platform's maximum (effective) alignment is 8 bytes, and that every integer
/// with a size in bytes of a power of two has the same alignment as its size.
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

/// Return a type that calls a closure when formatted using [Debug](std::fmt::Debug).
pub fn debug_with<F>(fmt: F) -> impl fmt::Debug
where
    F: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    struct FmtWrapper<F>(F)
    where
        F: Fn(&mut fmt::Formatter) -> fmt::Result;

    impl<F> fmt::Debug for FmtWrapper<F>
    where
        F: Fn(&mut fmt::Formatter) -> fmt::Result,
    {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            self.0(f)
        }
    }

    FmtWrapper(fmt)
}

#[cfg(test)]
mod test {
    use super::*;

    /// Ensures that the host's `u8` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u8() {
        assert_eq!(get_integer_layout(8).align(), 1);
    }

    /// Ensures that the host's `u16` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u16() {
        assert_eq!(get_integer_layout(16).align(), 2);
    }

    /// Ensures that the host's `u32` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u32() {
        assert_eq!(get_integer_layout(32).align(), 4);
    }

    /// Ensures that the host's `u64` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u64() {
        assert_eq!(get_integer_layout(64).align(), 8);
    }

    /// Ensures that the host's `u128` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u128() {
        assert_eq!(get_integer_layout(128).align(), 8);
    }

    /// Ensures that the host's `u256` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u256() {
        assert_eq!(get_integer_layout(256).align(), 8);
    }

    /// Ensures that the host's `u512` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u512() {
        assert_eq!(get_integer_layout(512).align(), 8);
    }

    /// Ensures that the host's `felt252` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_felt252() {
        assert_eq!(get_integer_layout(252).align(), 8);
    }
}
